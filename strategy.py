#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.8x average)
# Donchian channels provide clear structure for breakouts. EMA50 on 1d filters for higher timeframe trend.
# Volume confirmation ensures breakouts have conviction. Works in bull/bear via trend filter.
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_Donchian20_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 4h data
    # Use rolling window with min_periods
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or
            np.isnan(low_roll[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_upper = high_roll[i]
        curr_lower = low_roll[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Donchian breakout
            if curr_volume_spike:
                # Bullish: price breaks above Donchian upper + price above 1d EMA50
                if curr_high > curr_upper and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian lower + price below 1d EMA50
                elif curr_low < curr_lower and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower (trend reversal)
            if curr_low < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper (trend reversal)
            if curr_high > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals