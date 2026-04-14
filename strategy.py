#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points using prior day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point: (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    # Resistance and support levels
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Chop filter: avoid ranging markets
    # Calculate True Range for 12h period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: high values indicate ranging market
    # We'll use a simple version: if ATR is low relative to price, it's choppy
    # Normalize ATR by price
    atr_ratio = atr / close
    # Avoid division by zero or extreme values
    atr_ratio = np.where(close > 0, atr_ratio, 0)
    # Smooth the ratio
    atr_ratio_smooth = pd.Series(atr_ratio).rolling(window=10, min_periods=10).mean().values
    # Chop condition: avoid when ATR ratio is very low (ranging market)
    chop_filter = atr_ratio_smooth > 0.005  # Adjust threshold as needed
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume and ATR calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(avg_vol[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume confirmation and not in chop
            if price > r2_aligned[i] and vol > 1.5 * avg_vol[i] and chop_filter[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 with volume confirmation and not in chop
            elif price < s2_aligned[i] and vol > 1.5 * avg_vol[i] and chop_filter[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1
            if price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1
            if price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Pivot_Breakout_Volume_Chop"
timeframe = "12h"
leverage = 1.0