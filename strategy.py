#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (triple-smoothed EMA) momentum with volume spike confirmation and choppiness regime filter. 
Long when TRIX crosses above zero with volume spike in choppy/range market (CHOP > 61.8). 
Short when TRIX crosses below zero with volume spike in choppy/range market. 
Uses 12h EMA50 as trend filter to avoid counter-trend trades. 
Discrete sizing 0.25 to limit trades (~20-40/year). Works in bull/bear via 12h trend filter and chop regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # TRIX: triple EMA of close, then ROC
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change(periods=1))
    trix_values = trix.values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest(high,n) - lowest(low,n))))
    # Simplified: CHOP > 61.8 = range, CHOP < 38.2 = trend
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Sum of ATR over 14 periods
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (highest_high - lowest_low)))
    # Handle division by zero or invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    
    # Volume spike: volume > 2.0 * average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 12h EMA, 15*3 for TRIX, 14 for ATR/CHOP, 20 for volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(trix_values[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        trix_val = trix_values[i]
        trix_prev = trix_values[i-1]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # TRIX crossing zero with volume spike in choppy market
            trix_cross_up = trix_prev <= 0 and trix_val > 0
            trix_cross_down = trix_prev >= 0 and trix_val < 0
            
            # Only trade in choppy/range market (CHOP > 61.8)
            in_chop = chop_val > 61.8
            
            # Long entry: TRIX crosses up + volume spike + chop + above 12h EMA50 (uptrend bias)
            long_entry = trix_cross_up and vol_spike and in_chop and (close_val > ema_50_val)
            # Short entry: TRIX crosses down + volume spike + chop + below 12h EMA50 (downtrend bias)
            short_entry = trix_cross_down and vol_spike and in_chop and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TRIX cross down or trend change
            if trix_val < 0 or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on TRIX cross up or trend change
            if trix_val > 0 or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0