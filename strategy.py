#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use 12h price breaking above/below Camarilla R1/S1 levels from prior 1d session,
# in direction of 1w trend (EMA34 on 1w close), with volume confirmation (>1.5x 20-period avg volume).
# Camarilla levels provide precise intraday support/resistance; 1w trend filters for major trend direction;
# volume ensures breakout conviction. Designed for low trade frequency (<30/year) to avoid fee drag.
# Works in bull/bear by aligning with 1w trend direction.

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
    
    # 12h Camarilla levels from prior 1d session
    # Calculate using prior 1d OHLC (shifted by 1 to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC values
    phigh = df_1d['high'].shift(1).values  # prior day high
    plow = df_1d['low'].shift(1).values    # prior day low
    pclose = df_1d['close'].shift(1).values # prior day close
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = pclose + (phigh - plow) * 1.1 / 12
    camarilla_s1 = pclose - (phigh - plow) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for prior 1d close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1w trend filter: EMA(34) on 1w close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1, 1w trend up, volume confirmation
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1, 1w trend down, volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Camarilla S1 OR trend changes
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Camarilla R1 OR trend changes
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals