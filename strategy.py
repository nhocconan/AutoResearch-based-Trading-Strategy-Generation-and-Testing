#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Donchian breakouts capture momentum shifts; weekly pivot (from prior week) provides institutional bias
# Volume spike (>2.0 x 30-period EMA) confirms breakout validity
# Discrete position sizing (0.25) controls fee drag while maintaining exposure
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns
# Works in both bull and bear markets by requiring alignment with weekly trend

name = "6h_Donchian20_Breakout_1wPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_30)
    
    # Weekly data for Donchian channels and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous weekly bar (20-period)
    # Upper = max(high over last 20 weekly bars)
    # Lower = min(low over last 20 weekly bars)
    # Using shift(1) to ensure we only use completed weekly bars
    win = 20
    if len(df_1w) < win + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min on weekly data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Compute Donchian channels using pandas rolling for clarity
    dh_series = pd.Series(weekly_high).rolling(window=win, min_periods=win).max().shift(1).values
    dl_series = pd.Series(weekly_low).rolling(window=win, min_periods=win).min().shift(1).values
    
    # Align Donchian levels to 6h timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, dh_series)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, dl_series)
    
    # Weekly pivot points from previous weekly bar
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    weekly_pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pp - prev_weekly_low
    weekly_s1 = 2 * weekly_pp - prev_weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly bias from pivot levels
        weekly_bullish = close[i] > weekly_pp[i] if not np.isnan(weekly_pp[i]) else False
        weekly_bearish = close[i] < weekly_pp[i] if not np.isnan(weekly_pp[i]) else False
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian high with volume confirmation and weekly bullish bias
            if close[i] > donchian_high_aligned[i] and volume_confirmation[i] and weekly_bullish:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low with volume confirmation and weekly bearish bias
            elif close[i] < donchian_low_aligned[i] and volume_confirmation[i] and weekly_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian low OR weekly bias turns bearish
            if close[i] < donchian_low_aligned[i] or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian high OR weekly bias turns bullish
            if close[i] > donchian_high_aligned[i] or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals