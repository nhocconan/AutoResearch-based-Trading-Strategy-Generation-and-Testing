#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d weekly Donchian breakout with 1w EMA trend filter and volume confirmation
    # Long: price breaks above weekly Donchian high(20) AND weekly EMA(21) rising AND volume > 1.5x avg
    # Short: price breaks below weekly Donchian low(20) AND weekly EMA(21) falling AND volume > 1.5x avg
    # Exit: price touches opposite Donchian level OR weekly EMA crosses in opposite direction
    # Using 1d timeframe for optimal trade frequency (target 7-25/year), weekly Donchian for structure,
    # weekly EMA for trend filter, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian high and low (20-period)
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Calculate weekly EMA(21) for trend direction
    ema_21 = np.full_like(close_1w, np.nan)
    ema_21[20] = np.mean(close_1w[:21])  # Simple average for first value
    alpha = 2 / (21 + 1)
    for i in range(21, len(close_1w)):
        ema_21[i] = alpha * close_1w[i] + (1 - alpha) * ema_21[i-1]
    
    # Align weekly indicators to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Calculate daily volume MA(20) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA rising/falling
        ema_rising = ema_21_aligned[i] > ema_21_aligned[i-1]
        ema_falling = ema_21_aligned[i] < ema_21_aligned[i-1]
        
        # Donchian breakout conditions
        breakout_high = close[i] > donchian_high_aligned[i]
        breakout_low = close[i] < donchian_low_aligned[i]
        
        # Exit conditions: opposite Donchian touch OR EMA cross in opposite direction
        exit_long = close[i] < donchian_low_aligned[i] or (position == 1 and ema_falling)
        exit_short = close[i] > donchian_high_aligned[i] or (position == -1 and ema_rising)
        
        # Entry logic: Donchian breakout + EMA trend + volume confirmation
        long_entry = breakout_high and ema_rising and volume_spike[i]
        short_entry = breakout_low and ema_falling and volume_spike[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_ema21_volume_v1"
timeframe = "1d"
leverage = 1.0