#!/usr/bin/env python3
"""
6h_ADX_Alligator_Trend_Filter
Hypothesis: Combining ADX trend strength with Williams Alligator lines on 12h timeframe filters false signals.
In trending markets (ADX>25), price above/below Alligator jaws/teeth/lips indicates strong trend.
Works in both bull/bear by capturing strong trends while avoiding whipsaws in ranging markets.
Target: 15-30 trades/year (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    plus_di = 100 * (np.zeros(n))
    minus_di = 100 * (np.zeros(n))
    dx = np.zeros(n)
    
    # Smooth +DM and -DM
    plus_dm_sm = np.zeros(n)
    minus_dm_sm = np.zeros(n)
    plus_dm_sm[0] = plus_dm[0]
    minus_dm_sm[0] = minus_dm[0]
    for i in range(1, n):
        plus_dm_sm[i] = (plus_dm_sm[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_sm[i] = (minus_dm_sm[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DI and DX
    for i in range(n):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_sm[i] / atr[i]
            minus_di[i] = 100 * minus_dm_sm[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX: smoothed DX
    adx = np.zeros(n)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) SMMA
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    median_price_12h = (high_12h + low_12h) / 2
    
    def smma(arr, period):
        res = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return res
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    jaw = smma(median_price_12h, 13)
    teeth = smma(median_price_12h, 8)
    lips = smma(median_price_12h, 5)
    
    # Align Alligator lines to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_12h, teeth)
    lips_6h = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for ADX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or 
            np.isnan(lips_6h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        jaw_val = jaw_6h[i]
        teeth_val = teeth_6h[i]
        lips_val = lips_6h[i]
        vol_ok = volume_filter[i]
        
        # Alligator alignment: all three lines in order
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: strong uptrend (ADX>25) + bullish alignment + volume
            if adx_val > 25 and bullish_alignment and vol_ok and price > lips_val:
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend (ADX>25) + bearish alignment + volume
            elif adx_val > 25 and bearish_alignment and vol_ok and price < lips_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakens (ADX<20) or alignment breaks
            if adx_val < 20 or not bullish_alignment or price < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakens (ADX<20) or alignment breaks
            if adx_val < 20 or not bearish_alignment or price > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Alligator_Trend_Filter"
timeframe = "6h"
leverage = 1.0