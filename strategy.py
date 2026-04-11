#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness regime + 1w ATR breakout + volume confirmation
# Long when Choppiness < 38.2 (trending) + price breaks above ATR(14) upper band + volume > 2x average
# Short when Choppiness < 38.2 (trending) + price breaks below ATR(14) lower band + volume > 2x average
# Exit when Choppiness > 61.8 (range) or price returns to ATR midline
# Designed for 12-37 trades/year on 12h timeframe with strong trend capture in both bull/bear markets

name = "12h_1w_atr_breakout_choppiness_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range components
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # First period TR
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1w = np.zeros_like(tr)
    atr_1w[13] = np.mean(tr[:14])  # Initial ATR as average of first 14 TR
    for i in range(14, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR bands using 1w close as base
    atr_mult = 2.0
    atr_upper_1w = close_1w + atr_1w * atr_mult
    atr_lower_1w = close_1w - atr_1w * atr_mult
    atr_mid_1w = close_1w  # Midline is close
    
    # Align ATR bands to 12h timeframe
    atr_upper_aligned = align_htf_to_ltf(prices, df_1w, atr_upper_1w)
    atr_lower_aligned = align_htf_to_ltf(prices, df_1w, atr_lower_1w)
    atr_mid_aligned = align_htf_to_ltf(prices, df_1w, atr_mid_1w)
    
    # Load 1d data for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_d[0] = high_1d[0] - low_1d[0]
    tr2_d[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_d[0] = np.abs(low_1d[0] - close_1d[0])
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    
    # Sum of TR over 14 periods
    sum_tr_d = np.zeros_like(tr_d)
    for i in range(14, len(tr_d)):
        sum_tr_d[i] = np.sum(tr_d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high_d = np.zeros_like(high_1d)
    min_low_d = np.zeros_like(low_1d)
    for i in range(13, len(high_1d)):
        max_high_d[i] = np.max(high_1d[i-13:i+1])
        min_low_d[i] = np.min(low_1d[i-13:i+1])
    
    # Avoid division by zero
    range_d = max_high_d - min_low_d
    range_d[range_d == 0] = 1e-10
    
    # Choppiness Index formula
    chop = 100 * np.log10(sum_tr_d / range_d) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # Default to middle range
    
    # Align Choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_upper_aligned[i]) or np.isnan(atr_lower_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: Choppiness < 38.2 indicates trending market
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        # Entry conditions
        long_entry = is_trending and volume_filter and (close[i] > atr_upper_aligned[i-1])
        short_entry = is_trending and volume_filter and (close[i] < atr_lower_aligned[i-1])
        
        # Exit conditions
        long_exit = is_ranging or (close[i] < atr_mid_aligned[i])
        short_exit = is_ranging or (close[i] > atr_mid_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals