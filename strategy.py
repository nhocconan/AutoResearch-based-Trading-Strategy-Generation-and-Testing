#!/usr/bin/env python3
# 1d_1w_camarilla_volume_breakout_v1
# Hypothesis: Trade Camarilla pivot level breaks on daily timeframe with weekly trend filter and volume confirmation.
# In bullish weekly regime (price > weekly 50-period SMA): long when price breaks above daily Camarilla H4 level.
# In bearish weekly regime (price < weekly 50-period SMA): short when price breaks below daily Camarilla L4 level.
# Uses volume confirmation (1.8x average) to filter breakouts. Designed for low trade frequency (~15-25/year)
# to minimize fee drag while capturing strong directional moves in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_volume_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly 50-period SMA for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # Note: Using previous day's values to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to daily timeframe (already aligned as we're using 1d data)
    # But we still need to ensure proper alignment with the weekly trend
    camarilla_h4_aligned = camarilla_h4  # Already on daily scale
    camarilla_l4_aligned = camarilla_l4  # Already on daily scale
    
    # Volume confirmation: volume > 1.8x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure weekly SMA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price drops below Camarilla L4 level
            if close[i] < camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla H4 level
            if close[i] > camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above H4 with volume surge and bullish weekly trend
            if (close[i] > camarilla_h4_aligned[i] and vol_surge and 
                close[i] > sma50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume surge and bearish weekly trend
            elif (close[i] < camarilla_l4_aligned[i] and vol_surge and 
                  close[i] < sma50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals