//! Raymarching stuff
use std::any::Any;
use std::ops::RangeInclusive;

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bevy_egui::egui::plot::Plot;
use bevy_egui::egui::{Color32, Ui};
use bevy_egui::{egui, EguiContext};

use crate::widgets::{Curve, Widget};

// Raymarching related settings

#[derive(Copy, Clone)]
struct DensityFalloff(f64);

#[derive(ShaderType, Clone)]
pub struct SkySettings {
    planet_center: Vec3,
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_distance: f32,
    sun_axis: Vec3,
    density_falloff: f32,
    wavelengths: Vec3,
    scatter_strength: f32,
    scatter_coefficients: Vec3,
}

pub struct RaymarchPlugin;

impl Plugin for RaymarchPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SkySettings::default())
            .add_system(draw_settings::<SkySettings>);
    }
}

impl Default for SkySettings {
    fn default() -> Self {
        SkySettings {
            planet_center: Vec3::default(),
            planet_radius: 1.0,
            atmosphere_radius: 0.27,
            sun_distance: 150_000_000.0,
            sun_axis: Vec3::Y,
            density_falloff: 6.0,
            wavelengths: Vec3::new(700., 530., 440.),
            scatter_strength: 11.7,
            scatter_coefficients: Default::default(),
        }
    }
}

impl Widget for SkySettings {
    fn draw(&mut self, ui: &mut Ui) {
        ui.label("Earth");
        ui.add(egui::Slider::new(&mut self.planet_radius, 0.1..=2.).text("Radius"));
        ui.add(egui::Slider::new(&mut self.atmosphere_radius, 0.0001..=1.).text("Atmosphere"));

        ui.label("Density Falloff");
        ui.add(egui::Slider::new(&mut self.density_falloff, 0.1..=15.));
        let mut falloff = DensityFalloff(self.density_falloff as f64);
        falloff.draw(ui);

        ui.label("Scattering");
        ui.add(egui::Slider::new(&mut self.scatter_strength, 0.1..=15.).text("Scatter strength"));
        ui.add(egui::Slider::new(&mut self.wavelengths.x, 100.0..=800.).text("Red wavelength"));
        ui.add(egui::Slider::new(&mut self.wavelengths.y, 100.0..=800.).text("Green wavelength"));
        ui.add(egui::Slider::new(&mut self.wavelengths.z, 100.0..=800.).text("Blue wavelength"));

        self.scatter_coefficients.x = (400. / self.wavelengths.x).powf(4.0) * self.scatter_strength;
        self.scatter_coefficients.y = (400. / self.wavelengths.y).powf(4.0) * self.scatter_strength;
        self.scatter_coefficients.z = (400. / self.wavelengths.z).powf(4.0) * self.scatter_strength;

        let mut scatter_curves = ScatterCurves(self.scatter_coefficients);
        scatter_curves.draw(ui);
    }
}

impl Curve for DensityFalloff {
    const VALUE_RANGE: RangeInclusive<f64> = 0. ..=2.;

    fn value_at(&self, x: f64) -> f64 {
        (-x * self.0).exp()
    }

    fn customize_plot(&self, plot: Plot) -> Plot {
        plot.width(400.).height(100.)
    }
}

struct ScatterCurve(f64);

impl Curve for ScatterCurve {
    const VALUE_RANGE: RangeInclusive<f64> = 0.0..=3.0;

    fn value_at(&self, x: f64) -> f64 {
        (-x * self.0).exp()
    }
}

struct ScatterCurves(Vec3);

impl Widget for ScatterCurves {
    fn draw(&mut self, ui: &mut Ui) {
        let r_curve = ScatterCurve(self.0.x as f64);
        let g_curve = ScatterCurve(self.0.y as f64);
        let b_curve = ScatterCurve(self.0.z as f64);

        let r_line = r_curve.line().color(Color32::RED);
        let g_line = g_curve.line().color(Color32::GREEN);
        let b_line = b_curve.line().color(Color32::BLUE);

        Plot::new("ScatterCurves")
            .width(400.)
            .height(100.)
            .show(ui, |plot_ui| {
                plot_ui.line(r_line);
                plot_ui.line(g_line);
                plot_ui.line(b_line);
            });
    }
}

fn draw_settings<T>(mut settings: ResMut<T>, mut egui_context: ResMut<EguiContext>)
where
    T: Widget + Any + Send + Sync,
{
    egui::Window::new(std::any::type_name::<T>()).show(egui_context.ctx_mut(), |ui| {
        settings.draw(ui);
    });
}
