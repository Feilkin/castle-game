//! UI widgets and traits
use std::any::{Any, TypeId};
use std::ops::RangeInclusive;

use bevy::prelude::*;
use bevy_egui::egui::plot::{Line, Plot, PlotPoints};
use bevy_egui::egui::Ui;

pub trait Widget {
    fn draw(&mut self, ui: &mut Ui);
}

pub trait Curve {
    /// Allowed value range for this curve
    const VALUE_RANGE: RangeInclusive<f64>;
    const SAMPLE_COUNT: usize = 100;

    /// Value at a given point in this curve.
    fn value_at(&self, x: f64) -> f64;

    /// Optional hook to customize the plot before displaying it.
    fn customize_plot(&self, plot: Plot) -> Plot {
        plot
    }

    fn points(&self) -> PlotPoints {
        PlotPoints::from_parametric_callback(
            |x| (x, self.value_at(x)),
            Self::VALUE_RANGE,
            Self::SAMPLE_COUNT,
        )
    }

    fn line(&self) -> Line {
        Line::new(self.points())
    }
}

impl<T: Curve + Any> Widget for T {
    fn draw(&mut self, ui: &mut Ui) {
        let line = self.line();

        let plot = Plot::new(TypeId::of::<Self>());
        let plot = self.customize_plot(plot);

        plot.show(ui, |plot_ui| plot_ui.line(line));
    }
}
