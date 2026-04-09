#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Supertrend for trend direction and 1w Williams %R for extreme entry timing
# Supertrend(10, 3.0) on 1d provides reliable trend filter - long when price > Supertrend, short when price < Supertrend
# Williams %R(14) on 1w identifies overbought/oversold conditions - long when %R crosses above -80 from below,
# short when %R crosses below -20 from above. This combination provides trend-following with precise entries
# during pullbacks in the trend. Works in both bull and bear markets by adapting to the 1d trend.
# Discrete position sizing 0.25 limits trades to ~12-37/year and reduces fee drag.

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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(10) for Supertrend
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
    
    atr_1d = wilders_smoothing(tr, 10)
    
    # Calculate 1d Supertrend(10, 3.0)
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr_1d)
    lower_band = hl2 - (3.0 * atr_1d)
    
    upper_band_final = np.copy(upper_band)
    lower_band_final = np.copy(lower_band)
    supertrend = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Upper band
        if close_1d[i-1] <= upper_band_final[i-1]:
            upper_band_final[i] = min(upper_band[i], upper_band_final[i-1])
        else:
            upper_band_final[i] = upper_band[i]
        
        # Lower band
        if close_1d[i-1] >= lower_band_final[i-1]:
            lower_band_final[i] = max(lower_band[i], lower_band_final[i-1])
        else:
            lower_band_final[i] = lower_band[i]
        
        # Supertrend
        if i == 1:
            supertrend[i] = upper_band_final[i]
        else:
            if supertrend[i-1] == upper_band_final[i-1]:
                if close_1d[i] <= upper_band_final[i]:
                    supertrend[i] = upper_band_final[i]
                else:
                    supertrend[i] = lower_band_final[i]
            else:
                if close_1d[i] >= lower_band_final[i]:
                    supertrend[i] = lower_band_final[i]
                else:
                    supertrend[i] = upper_band_final[i]
    
    # Determine 1d trend direction: 1 = uptrend (price > Supertrend), -1 = downtrend (price < Supertrend)
    trend_1d = np.where(close_1d > supertrend, 1, -1)
    
    # Load 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R(14)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.full(len(high), np.nan)
        lowest_low = np.full(len(low), np.nan)
        
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-(period-1):i+1])
            lowest_low[i] = np.min(low[i-(period-1):i+1])
        
        williams_r = np.full(len(close), np.nan)
        for i in range(period-1, len(close)):
            if highest_high[i] - lowest_low[i] != 0:
                williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
            else:
                williams_r[i] = -50
        
        return williams_r
    
    williams_r_1w = calculate_williams_r(high_1w, low_1w, close_1w, 14)
    
    # Align 1d indicators to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Align 1w Williams %R to 6h timeframe
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(williams_r_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if trend turns down OR Williams %R becomes overbought (> -20)
            if trend_1d_aligned[i] == -1 or williams_r_1w_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit short if trend turns up OR Williams %R becomes oversold (< -80)
            if trend_1d_aligned[i] == 1 or williams_r_1w_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        
        else:  # Flat
            # Enter long: uptrend AND Williams %R crosses above -80 from below
            if (trend_1d_aligned[i] == 1 and 
                williams_r_1w_aligned[i] > -80 and 
                i > 100 and 
                williams_r_1w_aligned[i-1] <= -80):
                position = 1
                signals[i] = 0.25
            
            # Enter short: downtrend AND Williams %R crosses below -20 from above
            elif (trend_1d_aligned[i] == -1 and 
                  williams_r_1w_aligned[i] < -20 and 
                  i > 100 and 
                  williams_r_1w_aligned[i-1] >= -20):
                position = -1
                signals[i] = -0.25
    
    return signals