#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Supertrend (ATR=10, mult=3.0) for trend direction and 1d Camarilla pivot levels (R3/S3) for mean reversion entries.
# In 1w uptrend (close > Supertrend), wait for price to touch or breach 1d S3 level to go long (mean reversion in uptrend).
# In 1w downtrend (close < Supertrend), wait for price to touch or breach 1d R3 level to go short (mean reversion in downtrend).
# Volume confirmation ensures momentum validity. Designed for low trade frequency (12-30/year) to minimize fee drag while adapting to trend and mean reversion.
# Uses weekly Supertrend to avoid whipsaw in sideways markets and Camarilla levels for precise mean reversion entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1w and 1d HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Supertrend (ATR=10, mult=3.0) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # first bar
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upperband and Lowerband
    basic_ub = (high_1w + low_1w) / 2 + 3.0 * atr_1w
    basic_lb = (high_1w + low_1w) / 2 - 3.0 * atr_1w
    
    # Final Upperband and Lowerband
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_1w)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
            supertrend[i] = final_ub[i]
            direction[i] = 1
        else:
            if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
            
            if supertrend[i-1] == final_ub[i-1]:
                if close_1w[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lb[i]
                    direction[i] = 1
            else:
                if close_1w[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_ub[i]
                    direction[i] = -1
    
    # Supertrend values (only the indicator line, not direction)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 4.0
    s3_1d = pivot_1d - range_1d * 1.1 / 4.0
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1w uptrend (close > Supertrend)
        # 2. Price touches or breaches 1d S3 level (mean reversion long)
        # 3. Volume confirmation
        if (close[i] > supertrend_aligned[i]) and (low[i] <= s3_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1w downtrend (close < Supertrend)
        # 2. Price touches or breaches 1d R3 level (mean reversion short)
        # 3. Volume confirmation
        elif (close[i] < supertrend_aligned[i]) and (high[i] >= r3_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Supertrend1w_Camarilla1d_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0