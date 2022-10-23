//! Building types
use bevy::prelude::*;

#[derive(Component)]
pub struct Tower {
    pub(crate) height: f32,
}

impl Tower {
    pub const KIND: i32 = 1;
}
