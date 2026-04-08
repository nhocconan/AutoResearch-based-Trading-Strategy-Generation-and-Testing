#!/usr/bin/env python3
"""
6h_weekly_trend_following_v1
Hypothesis: Follow weekly trend using price relative to weekly VWAP with ADX filter.
- Only trade in direction of weekly trend (price above/below weekly VWAP)
- Use ADX(14) on 6h to filter for trending conditions (ADX > 25)
- Enter on pullbacks to weekly VWAP in trending markets
- Exit when trend weakens (ADX < 20) or price crosses weekly VWAP
- Target: 20-40 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_trend_following_v1"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly VWAP calculation
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_num = (typical_price_1w * df_1w['volume']).cumsum()
    vwap_den = df_1w['volume'].cumsum()
    weekly_vwap = (vwap_num / vwap_den).values
    
    # Weekly trend: price relative to VWAP
    weekly_price_above_vwap = df_1w['close'].values > weekly_vwap
    weekly_price_below_vwap = df_1w['close'].values < weekly_vwap
    
    # Align weekly trend to 6h
    weekly_above_aligned = align_htf_to_ltf(prices, df_1w, weekly_price_above_vwap.astype(float))
    weekly_below_aligned = align_htf_to_ltf(prices, df_1w, weekly_price_below_vwap.astype(float))
    
    # ADX calculation on 6h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
            minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Smoothed DX
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Warmup period
        # Skip if data not ready
        if (np.isnan(weekly_above_aligned[i]) or np.isnan(weekly_below_aligned[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: trend weakens or price crosses below weekly VWAP
            if adx[i] < 20 or weekly_below_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: trend weakens or price crosses above weekly VWAP
            if adx[i] < 20 or weekly_above_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: weekly uptrend + ADX trending + pullback to VWAP
            if (weekly_above_aligned[i] > 0.5 and adx[i] > 25 and 
                close[i] <= weekly_vwap[i] * 1.005 and  # Near VWAP (within 0.5%)
                close[i-1] > weekly_vwap[i-1]):  # Was above, now pulling back
                position = 1
                signals[i] = 0.25
            # Enter short: weekly downtrend + ADX trending + pullback to VWAP
            elif (weekly_below_aligned[i] > 0.5 and adx[i] > 25 and 
                  close[i] >= weekly_vwap[i] * 0.995 and  # Near VWAP (within 0.5%)
                  close[i-1] < weekly_vwap[i-1]):  # Was below, now pulling back
                position = -1
                signals[i] = -0.25
    
    return signals