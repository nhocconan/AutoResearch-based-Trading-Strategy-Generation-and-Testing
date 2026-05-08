#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Exponential Moving Average (50) trend filter with weekly Donchian breakout and volume confirmation
# Long when daily close crosses above 50 EMA and weekly price breaks above 10-period Donchian high with volume spike
# Short when daily close crosses below 50 EMA and weekly price breaks below 10-period Donchian low with volume spike
# EMA provides smooth trend direction, Donchian captures breakouts in higher timeframe, volume confirms validity
# Targets 20-50 total trades over 4 years (5-12/year) to minimize fee drag

name = "1d_EMA50_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Donchian breakout
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly 10-period Donchian channels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    donchian_high = pd.Series(weekly_high).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=10, min_periods=10).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate daily EMA(50) for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        weekly_close_val = weekly_close[i // 7] if i // 7 < len(weekly_close) else weekly_close[-1]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: daily close crosses above EMA50 and weekly close breaks above Donchian high with volume spike
            if close[i] > ema50_val and close[i-1] <= ema50_val and weekly_close_val > donchian_high_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: daily close crosses below EMA50 and weekly close breaks below Donchian low with volume spike
            elif close[i] < ema50_val and close[i-1] >= ema50_val and weekly_close_val < donchian_low_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: daily close crosses below EMA50
            if close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: daily close crosses above EMA50
            if close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals