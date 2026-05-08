#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Supertrend for trend direction with 4h volume confirmation and ATR-based volatility filter.
# Supertrend (ATR=10, multiplier=3) on daily timeframe provides robust trend identification.
# Entry only in trend direction when 4h volume exceeds 2x its 20-period average.
# Exit when price closes below/above Supertrend or when volatility drops (ATR < 0.5 * ATR(20)).
# Designed for low trade frequency (<25/year) to minimize fee drag while capturing major trends.
# Works in both bull and bear markets by following the trend direction.

name = "4h_Supertrend_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for Supertrend (period=10)
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
    
    atr = np.zeros(len(high_1d))
    atr[:9] = np.nan
    atr[9] = np.mean(tr1[0:10])
    for i in range(10, len(high_1d)):
        atr[i] = (atr[i-1] * 9 + tr1[i]) / 10
    
    # Calculate Supertrend
    upper_band = np.zeros(len(high_1d))
    lower_band = np.zeros(len(high_1d))
    supertrend = np.zeros(len(high_1d))
    trend = np.ones(len(high_1d))  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(high_1d)):
        if i < 9:
            upper_band[i] = np.nan
            lower_band[i] = np.nan
            supertrend[i] = np.nan
            trend[i] = np.nan
        else:
            upper_band[i] = (high_1d[i] + low_1d[i]) / 2 + 3 * atr[i]
            lower_band[i] = (high_1d[i] + low_1d[i]) / 2 - 3 * atr[i]
            
            if i == 9:
                supertrend[i] = upper_band[i]
                trend[i] = -1  # start in downtrend
            else:
                if close_1d[i-1] > upper_band[i-1]:
                    trend[i] = 1
                elif close_1d[i-1] < lower_band[i-1]:
                    trend[i] = -1
                else:
                    trend[i] = trend[i-1]
                
                if trend[i] == 1:
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
                else:
                    supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # 4h volume confirmation: volume > 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # 4h ATR for volatility filter (period=14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr_4h = np.zeros(n)
    atr_4h[:13] = np.nan
    atr_4h[13] = np.mean(tr[0:14])
    for i in range(14, n):
        atr_4h[i] = (atr_4h[i-1] * 13 + tr[i]) / 14
    
    # ATR volatility filter: current ATR > 0.5 * ATR(20)
    atr_ma = pd.Series(atr_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = (~np.isnan(atr_4h)) & (~np.isnan(atr_ma)) & (atr_4h > 0.5 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]) or
            np.isnan(atr_4h[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: uptrend + volume confirmation + volatility filter
            if trend_aligned[i] == 1 and vol_confirm[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + volume confirmation + volatility filter
            elif trend_aligned[i] == -1 and vol_confirm[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns down OR volatility drops
            if trend_aligned[i] == -1 or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns up OR volatility drops
            if trend_aligned[i] == 1 or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals