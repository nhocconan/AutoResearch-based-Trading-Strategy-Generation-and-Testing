#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above 20-day high AND 1w EMA34 is rising AND volume > 1.5x 20-day average volume.
Short when price breaks below 20-day low AND 1w EMA34 is falling AND volume > 1.5x 20-day average volume.
Exit when price touches the opposite Donchian band or after 10 bars (time-based exit).
Uses 1d for execution and Donchian bands, 1w for trend filter.
Designed to capture medium-term trends with volume confirmation and trend filter to avoid whipsaws.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian bands and volume MA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian bands (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # EMA34 slope: rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_rising[1:] = ema_34_1w[1:] > ema_34_1w[:-1]
    ema_34_falling[1:] = ema_34_1w[1:] < ema_34_1w[:-1]
    
    # Align all indicators to 1d timeframe (primary timeframe)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising.astype(float))
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    bars_in_trade = 0  # time-based exit counter
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_34_rising_aligned[i]) or
            np.isnan(ema_34_falling_aligned[i])):
            signals[i] = 0.0
            bars_in_trade = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: EMA34 rising for long, falling for short
        trend_long = ema_34_rising_aligned[i] > 0.5
        trend_short = ema_34_falling_aligned[i] > 0.5
        
        # Breakout conditions
        breakout_up = close[i] > donch_high_aligned[i]
        breakout_down = close[i] < donch_low_aligned[i]
        
        # Exit conditions: touch opposite band or time-based exit (10 bars)
        touch_opposite = (position == 1 and close[i] < donch_low_aligned[i]) or \
                         (position == -1 and close[i] > donch_high_aligned[i])
        time_exit = bars_in_trade >= 10
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and rising EMA34
            if (breakout_up and volume_confirmed and trend_long):
                signals[i] = 0.25
                position = 1
                bars_in_trade = 0
            # Short: break below Donchian low with volume confirmation and falling EMA34
            elif (breakout_down and volume_confirmed and trend_short):
                signals[i] = -0.25
                position = -1
                bars_in_trade = 0
        
        elif position == 1:
            # Exit long: touch Donchian low or time-based exit
            if (touch_opposite or time_exit):
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
                bars_in_trade += 1
        
        elif position == -1:
            # Exit short: touch Donchian high or time-based exit
            if (touch_opposite or time_exit):
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
                bars_in_trade += 1
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Trend"
timeframe = "1d"
leverage = 1.0