#!/usr/bin/env python3
"""
4h_1d_TRIX_Volume_Regime_v1
Hypothesis: TRIX momentum on 4h with volume spike confirmation and 1d chop regime filter.
Long when TRIX crosses above zero with volume spike in trending market (CHOP < 38.2).
Short when TRIX crosses below zero with volume spike in trending market.
Uses 1d Chop index to filter ranging markets (CHOP > 61.8) where momentum fails.
Designed for low trade frequency by requiring momentum confirmation, volume spike,
and regime alignment. Works in bull via momentum longs, in bear via momentum shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TRIX_Volume_Regime_v1"
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
    
    # === DAILY DATA FOR CHOP REGIME ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Chop index
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros(len(close_arr))
        tr = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = abs(high_arr[i] - close_arr[i-1])
            lc = abs(low_arr[i] - close_arr[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Smoothed TR (using simple average for simplicity)
        tr_sum = np.zeros(len(close_arr))
        tr_sum[0] = tr[0]
        for i in range(1, len(tr)):
            tr_sum[i] = tr_sum[i-1] + tr[i]
            if i >= period:
                tr_sum[i] -= tr[i-period]
        
        atr = tr_sum / period
        
        # True range of high-low over period
        max_high = np.zeros(len(close_arr))
        min_low = np.zeros(len(close_arr))
        for i in range(len(close_arr)):
            if i < period:
                max_high[i] = np.max(high_arr[:i+1])
                min_low[i] = np.min(low_arr[:i+1])
            else:
                max_high[i] = np.max(high_arr[i-period+1:i+1])
                min_low[i] = np.min(low_arr[i-period+1:i+1])
        
        # Avoid division by zero
        range_hl = max_high - min_low
        chop = np.zeros(len(close_arr))
        for i in range(len(close_arr)):
            if atr[i] > 0 and range_hl[i] > 0:
                chop[i] = 100 * np.log10(range_hl[i] / (atr[i] * period)) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4H TRIX CALCULATION ===
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1
    # Using 12-period as standard
    period = 12
    
    # First EMA
    ema1 = np.zeros(n)
    alpha = 2 / (period + 1)
    ema1[0] = close[0]
    for i in range(1, n):
        ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i-1]
    
    # Second EMA
    ema2 = np.zeros(n)
    ema2[0] = ema1[0]
    for i in range(1, n):
        ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
    
    # Third EMA
    ema3 = np.zeros(n)
    ema3[0] = ema2[0]
    for i in range(1, n):
        ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
    
    # TRIX calculation
    trix = np.zeros(n)
    trix[0] = 0
    for i in range(1, n):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
        else:
            trix[i] = 0
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for TRIX
        # Skip if not ready
        if (np.isnan(chop_1d_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_market = chop_1d_aligned[i] < 38.2
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # TRIX signals: zero cross
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Entry logic
        long_entry = trix_cross_up and trending_market and vol_confirm
        short_entry = trix_cross_down and trending_market and vol_confirm
        
        # Exit on opposite TRIX cross
        long_exit = trix_cross_down
        short_exit = trix_cross_up
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif long_exit and position == 1:
            position = 0
            signals[i] = 0.0
        elif short_exit and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals