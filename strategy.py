#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# The strategy trades long when price breaks above the 20-day high with weekly uptrend and high volume,
# and short when price breaks below the 20-day low with weekly downtrend and high volume.
# This captures breakouts in both bull and bear markets by aligning with the higher timeframe trend.
# Volume confirms the strength of the breakout. Target: 10-25 trades/year.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period average volume
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 49  # Need 20 days for Donchian + 50 for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_avg = vol_avg_20[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_current > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above Donchian high with weekly uptrend and volume confirmation
            if price > donchian_high[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with weekly downtrend and volume confirmation
            elif price < donchian_low[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below Donchian low or trend change
            if price < donchian_low[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above Donchian high or trend change
            if price > donchian_high[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals