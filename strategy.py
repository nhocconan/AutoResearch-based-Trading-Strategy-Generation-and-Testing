#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w volume confirmation and 1d trend filter
# - Long when price breaks above 20-period high + volume > 1.5x 1w average + price > 1d EMA50
# - Short when price breaks below 20-period low + volume > 1.5x 1w average + price < 1d EMA50
# - Exit on opposite Donchian break or trend reversal
# - Designed to capture strong trends with volume confirmation in both bull and bear markets
# - Target: 15-30 trades/year to minimize fee drag

name = "12h_Donchian20_1wVolume_1dTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # 1w volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1w average volume (scaled)
        # Scale 1w average to 12h: 1w has 14x 12h bars (7 days * 2 per day), so divide by 14
        volume_filter = vol_ma_1w_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1w_aligned[i] / 14.0)
        
        if position == 0:
            # Look for long entry: price > 20-period high + uptrend (price > 1d EMA50) + volume
            if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price < 20-period low + downtrend (price < 1d EMA50) + volume
            elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on price < 20-period low or trend reversal
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on price > 20-period high or trend reversal
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals