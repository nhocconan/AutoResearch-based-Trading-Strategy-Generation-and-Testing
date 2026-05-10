#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Breakout above/below Camarilla R1/S1 levels on 12h, filtered by 1w EMA50 trend and volume confirmation (>1.5x average).
# Uses ATR-based stoploss. Designed for 15-30 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate ATR(10) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(10, n):
        atr[i] = np.nanmean(tr[i-9:i+1])
    
    # Get 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Camarilla levels (based on previous 12h bar)
    close_12h = np.full(n, np.nan)
    high_12h = np.full(n, np.nan)
    low_12h = np.full(n, np.nan)
    for i in range(1, n):
        close_12h[i] = close[i-1]
        high_12h[i] = high[i-1]
        low_12h[i] = low[i-1]
    
    # Calculate Camarilla R1 and S1 from previous bar
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isnan(close_12h[i]) or np.isnan(high_12h[i]) or np.isnan(low_12h[i])):
            camarilla_r1[i] = close_12h[i] + (high_12h[i] - low_12h[i]) * 1.1 / 12
            camarilla_s1[i] = close_12h[i] - (high_12h[i] - low_12h[i]) * 1.1 / 12
    
    # Volume average (10 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(10, n):
        vol_ma[i] = np.nanmean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1w EMA50 trend
            if close[i] > ema_50_1w_aligned[i]:  # Uptrend
                # Long: Breakout above Camarilla R1 with volume confirmation
                if close[i] > camarilla_r1[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Camarilla S1 with volume confirmation
                if close[i] < camarilla_s1[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA50 or stoploss hit
            if close[i] < ema_50_1w_aligned[i] or (i > 0 and low[i] < camarilla_s1[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA50 or stoploss hit
            if close[i] > ema_50_1w_aligned[i] or (i > 0 and high[i] > camarilla_r1[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals