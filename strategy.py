#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-day high + price above 1w EMA50 + volume > 1.5x 20-day average
# - Short when price breaks below 20-day low + price below 1w EMA50 + volume > 1.5x 20-day average
# - Exit on opposite Donchian break (10-day) or trend reversal
# - Designed to capture trends in both bull and bear markets by following higher timeframe trend
# - Target: 10-25 trades/year to minimize fee drag

name = "1d_Donchian20_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period for entry, 10-period for exit)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max()
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min()
    
    # 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: uptrend + breakout + volume
            if close[i] > highest_high_20[i] and close[i] > ema_50_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend + breakdown + volume
            elif close[i] < lowest_low_20[i] and close[i] < ema_50_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown (10-day) or trend reversal
            if close[i] < lowest_low_10[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on breakout (10-day) or trend reversal
            if close[i] > highest_high_10[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals