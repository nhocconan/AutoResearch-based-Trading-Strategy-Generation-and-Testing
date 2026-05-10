#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout above/below Camarilla R1/S1 levels (1d) filtered by 1d EMA34 trend and volume confirmation (>1.5x average).
# Uses ATR-based stoploss. Designed for 20-50 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    cam_high = np.full(n, np.nan)
    cam_low = np.full(n, np.nan)
    cam_close = np.full(n, np.nan)
    for i in range(1, n):
        cam_high[i] = df_1d['high'].values[i-1] if i-1 < len(df_1d) else np.nan
        cam_low[i] = df_1d['low'].values[i-1] if i-1 < len(df_1d) else np.nan
        cam_close[i] = df_1d['close'].values[i-1] if i-1 < len(df_1d) else np.nan
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isnan(cam_high[i]) or np.isnan(cam_low[i]) or np.isnan(cam_close[i])):
            camarilla_r1[i] = cam_close[i] + (cam_high[i] - cam_low[i]) * 1.1 / 12
            camarilla_s1[i] = cam_close[i] - (cam_high[i] - cam_low[i]) * 1.1 / 12
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1d EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend
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
            # Exit: Price closes below EMA34 or stoploss hit
            if close[i] < ema_34_1d_aligned[i] or (i > 0 and low[i] < camarilla_s1[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA34 or stoploss hit
            if close[i] > ema_34_1d_aligned[i] or (i > 0 and high[i] > camarilla_r1[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals