#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (>1.8x average)
# Donchian channel provides clear structure. Breakout with volume and 1d trend filter reduces false signals.
# Works in bull/bear via trend filter. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_Donchian20_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20, 30)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Donchian breakout
            if curr_volume_spike:
                # Bullish: price breaks above Donchian high + price above 1d EMA34
                if curr_high > curr_donchian_high and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low + price below 1d EMA34
                elif curr_low < curr_donchian_low and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low (trend reversal)
            if curr_low < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (trend reversal)
            if curr_high > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals