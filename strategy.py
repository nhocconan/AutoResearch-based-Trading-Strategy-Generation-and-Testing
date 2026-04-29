#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h EMA trend + volume confirmation
# Donchian(20) breakout captures momentum; 12h EMA50 filters trend direction
# Volume spike (>1.5x 20-period MA) confirms breakout strength
# Works in both bull and bear markets by only trading breakouts in trend direction
# Target: 20-50 trades/year (80-200 total over 4 years)

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 (trend filter)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 12h EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper + above 12h EMA50 + volume confirmation
            if curr_close > curr_upper and curr_close > curr_ema_12h and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + below 12h EMA50 + volume confirmation
            elif curr_close < curr_lower and curr_close < curr_ema_12h and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when price breaks below Donchian lower
            if curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price breaks above Donchian upper
            if curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals