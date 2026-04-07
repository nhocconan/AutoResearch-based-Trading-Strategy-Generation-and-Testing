#!/usr/bin/env python3
"""
6h_supertrend_1w_trend_volume_v1
Hypothesis: On 6-hour timeframe, use Supertrend (ATR=10, multiplier=3) from weekly timeframe for trend direction with volume confirmation on 6H. 
Enter long when Supertrend turns bullish (green) and volume > 1.5x average, short when Supertrend turns bearish (red) and volume > 1.5x average. 
Exit when Supertrend flips direction. Uses weekly trend to avoid 6H whipsaws and volume to confirm momentum. Designed for low frequency (12-30 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on weekly data
    wt_high = df_1w['high'].values
    wt_low = df_1w['low'].values
    wt_close = df_1w['close'].values
    
    # ATR calculation
    atr_period = 10
    tr1 = wt_high[1:] - wt_low[1:]
    tr2 = np.abs(wt_high[1:] - wt_close[:-1])
    tr3 = np.abs(wt_low[1:] - wt_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    multiplier = 3
    hl_avg = (wt_high + wt_low) / 2
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr
    
    supertrend = np.full_like(wt_close, np.nan, dtype=float)
    trend = np.full_like(wt_close, 1, dtype=int)  # 1=up, -1=down
    
    for i in range(1, len(wt_close)):
        if np.isnan(atr[i]) or np.isnan(atr[i-1]):
            supertrend[i] = np.nan
            continue
            
        if wt_close[i] > upper_band[i-1]:
            trend[i] = 1
        elif wt_close[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and trend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    # Calculate 50-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after volume average warmup
        # Skip if weekly data not available
        if np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 50-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when Supertrend turns bearish
            if trend_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Supertrend turns bullish
            if trend_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Supertrend bullish AND volume confirmation
            long_entry = (trend_aligned[i] == 1) and vol_confirm
            # Short entry: Supertrend bearish AND volume confirmation
            short_entry = (trend_aligned[i] == -1) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals