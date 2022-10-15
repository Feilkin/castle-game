//! NPC and Task related things
use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use std::collections::VecDeque;

pub const MAX_DUDES: usize = 100;

pub struct NpcPlugin;

#[derive(Component)]
struct Npc;

#[derive(Debug)]
enum Task {
    Idle(Timer),
    MoveTo(Vec2),
}

#[derive(Debug, Component, Default)]
struct TaskQueue {
    tasks: VecDeque<Task>,
    fresh_task: bool,
}

fn do_tasks(mut entities: Query<(Entity, &mut Transform, &mut TaskQueue)>, time: Res<Time>) {
    for (_entity, mut transform, mut task_queue) in entities.iter_mut() {
        let mut task_finished = false;
        let fresh_task = task_queue.fresh_task;
        if let Some(task) = task_queue.tasks.front_mut() {
            let task: &mut Task = task;

            match task {
                Task::Idle(timer) => {
                    if fresh_task {
                        // animation.animation = "idle".to_string();
                    }

                    if timer.tick(time.delta()).finished() {
                        task_finished = true;
                    }
                }
                Task::MoveTo(target) => {
                    if fresh_task {
                        // animation.animation = "walk".to_string();
                    }

                    let distance = transform.translation.xy().distance(*target).min(1.0)
                        * time.delta_seconds()
                        * 10.;
                    let direction: Vec2 =
                        (*target - transform.translation.xy()).normalize_or_zero();
                    transform.translation += (direction * distance).extend(0.);
                    transform.rotation = Quat::from_rotation_z(direction.x.atan2(direction.y));

                    if distance <= 0.01 {
                        task_finished = true;
                    }
                }
            }
        }

        if task_finished {
            task_queue.tasks.pop_front();
            task_queue.fresh_task = true;
        } else {
            task_queue.fresh_task = false;
        }
    }
}

fn add_npc_tasks(mut npcs: Query<(&mut TaskQueue,), With<Npc>>) {
    for (mut task_queue,) in npcs.iter_mut() {
        if task_queue.tasks.is_empty() {
            let random_x = rand::random::<f32>() * 100. + 10.;
            let random_y = rand::random::<f32>() * 100.;
            let random_dur = rand::random::<f32>() * 3. + 1.5;
            task_queue
                .tasks
                .push_back(Task::MoveTo(Vec2::new(random_x, random_y)));
            task_queue
                .tasks
                .push_back(Task::Idle(Timer::from_seconds(random_dur, false)));
        }
    }
}
