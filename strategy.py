#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above 4h Donchian upper band + 1d volume > 2x average + price above 1w EMA50
# - Short when price breaks below 4h Donchian lower band + 1d volume > 2x average + price below 1w EMA50
# - Exit when price returns to 4h Donchian midpoint or trend reverses
# - Position size: 0.25 to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian(20) channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 2x 1d average volume (scaled to 4h)
        # 1d volume represents ~6 4h bars, so we compare to 1/6 of daily average
        vol_threshold = vol_ma_1d_aligned[i] / 6.0
        volume_filter = volume[i] > 2.0 * vol_threshold
        
        if position == 0:
            # Look for long entry: price breaks above Donchian high + volume + uptrend
            if close[i] > donch_high_aligned[i] and volume_filter and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below Donchian low + volume + downtrend
            elif close[i] < donch_low_aligned[i] and volume_filter and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to midpoint or trend reverses
            if close[i] < donch_mid_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to midpoint or trend reverses
            if close[i] > donch_mid_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals