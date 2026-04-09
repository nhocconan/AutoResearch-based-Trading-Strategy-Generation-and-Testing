#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Supertrend for trend direction and 1w Williams %R for mean reversion entries
# Supertrend(10,3) on 1d determines trend: price > Supertrend = uptrend, price < Supertrend = downtrend
# Williams %R(14) on 1w identifies overbought/oversold: > -20 = overbought, < -80 = oversold
# In uptrend: long when weekly Williams %R < -80 (oversold pullback), exit when > -20
# In downtrend: short when weekly Williams %R > -20 (overbought bounce), exit when < -80
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: buys dips in uptrend, sells rallies in downtrend

name = "6h_1d_1w_supertrend_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(10)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_10 = wilders_smoothing(tr, 10)
    
    # Calculate 1d Supertrend(10,3)
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + 3 * atr_10
    lower_band = hl2 - 3 * atr_10
    
    upper_band_final = np.copy(upper_band)
    lower_band_final = np.copy(lower_band)
    
    for i in range(1, len(close_1d)):
        if close_1d[i-1] > upper_band_final[i-1]:
            upper_band_final[i] = upper_band[i]
        else:
            upper_band_final[i] = min(upper_band[i], upper_band_final[i-1])
        
        if close_1d[i-1] < lower_band_final[i-1]:
            lower_band_final[i] = lower_band[i]
        else:
            lower_band_final[i] = max(lower_band[i], lower_band_final[i-1])
    
    supertrend = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            supertrend[i] = upper_band_final[i]
        else:
            if supertrend[i-1] == upper_band_final[i-1]:
                supertrend[i] = upper_band_final[i] if close_1d[i] <= upper_band_final[i] else lower_band_final[i]
            else:
                supertrend[i] = lower_band_final[i] if close_1d[i] >= lower_band_final[i] else upper_band_final[i]
    
    # Trend direction: 1 = uptrend (price > Supertrend), -1 = downtrend (price < Supertrend)
    trend_1d = np.where(close_1d > supertrend, 1, -1)
    
    # Load 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R(14)
    highest_high = np.zeros(len(high_1w))
    lowest_low = np.zeros(len(low_1w))
    
    for i in range(len(high_1w)):
        if i < 13:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high_1w[i-13:i+1])
            lowest_low[i] = np.min(low_1w[i-13:i+1])
    
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close_1w) / (highest_high - lowest_low), 
                          -50)
    
    # Align indicators to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(williams_r_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long when Williams %R > -20 (overbought)
            if williams_r_1w_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit short when Williams %R < -80 (oversold)
            if williams_r_1w_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        
        else:  # Flat
            # In uptrend: look for oversold entry (Williams %R < -80)
            # In downtrend: look for overbought entry (Williams %R > -20)
            if trend_1d_aligned[i] == 1:  # Uptrend
                if williams_r_1w_aligned[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif trend_1d_aligned[i] == -1:  # Downtrend
                if williams_r_1w_aligned[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals