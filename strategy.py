#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout above/below Camarilla R1/S1 levels (calculated from 1d OHLC) on 4h, filtered by 1d EMA34 trend and volume confirmation (>1.5x average).
# Uses ATR-based stoploss. Designed for 20-50 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get 1d OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 from 1d OHLC
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r1[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 12
            camarilla_s1[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 12
    
    # Align Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1d EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend
                # Long: Breakout above Camarilla R1 with volume confirmation
                if close[i] > camarilla_r1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Camarilla S1 with volume confirmation
                if close[i] < camarilla_s1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA34 or stoploss hit
            if close[i] < ema_34_1d_aligned[i] or (i > 0 and low[i] < camarilla_s1_aligned[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA34 or stoploss hit
            if close[i] > ema_34_1d_aligned[i] or (i > 0 and high[i] > camarilla_r1_aligned[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals