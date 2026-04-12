#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_convergence_divergence"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA convergence/divergence from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_fast = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_diff = ema_fast - ema_slow
    ema_diff_prev = np.roll(ema_diff, 1)
    ema_diff_prev[0] = np.nan
    ema_diff_accel = ema_diff - ema_diff_prev
    ema_diff_accel_prev = np.roll(ema_diff_accel, 1)
    ema_diff_accel_prev[0] = np.nan
    ema_diff_accel_aligned = align_htf_to_ltf(prices, df_1d, ema_diff_accel)
    
    # 4h price position relative to EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    price_above_ema50 = close > ema_50
    
    # Volume filter: current volume > 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if np.isnan(ema_diff_accel_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        bullish_momentum = ema_diff_accel_aligned[i] > 0
        bearish_momentum = ema_diff_accel_aligned[i] < 0
        
        long_signal = bullish_momentum and price_above_ema50[i] and volume_filter[i]
        short_signal = bearish_momentum and not price_above_ema50[i] and volume_filter[i]
        
        # Exit when momentum changes
        exit_long = ema_diff_accel_aligned[i] <= 0
        exit_short = ema_diff_accel_aligned[i] >= 0
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals