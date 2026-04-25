#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction filter (avoid counter-trend trades).
- Donchian Channel: 20-period high/low breakouts for momentum capture.
- Volume Filter: Current 1d volume > 1.5 * 20-period average 1d volume.
- Entry: Long when close > upper Donchian AND 1w EMA50 up AND volume confirmation.
         Short when close < lower Donchian AND 1w EMA50 down AND volume confirmation.
- Exit: Opposite Donchian break or EMA50 trend reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong trends while filtering chop and false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # EMA50 calculation on weekly close
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 20-period Donchian channels on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Donchian upper/lower bands (20-period high/low)
    donch_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (waits for 1d bar close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate 1d volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        donch_high_level = donch_high_aligned[i]
        donch_low_level = donch_low_aligned[i]
        ema_50_level = ema_50_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above = curr_close > donch_high_level
        broke_below = curr_close < donch_low_level
        
        # EMA50 trend direction
        ema_trend_up = curr_close > ema_50_level
        ema_trend_down = curr_close < ema_50_level
        
        # Exit conditions
        if position != 0:
            # Exit long: close breaks below lower Donchian OR EMA50 trend turns down
            if position == 1:
                if broke_below or ema_trend_down:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper Donchian OR EMA50 trend turns up
            elif position == -1:
                if broke_above or ema_trend_up:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with EMA50 trend filter and volume confirmation
        if position == 0:
            # Long: break above upper Donchian AND EMA50 trend up AND volume confirmation
            long_condition = broke_above and ema_trend_up and volume_confirm
            
            # Short: break below lower Donchian AND EMA50 trend down AND volume confirmation
            short_condition = broke_below and ema_trend_down and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0