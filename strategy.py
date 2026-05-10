#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS_Refined
# Hypothesis: Uses Camarilla R1/S1 levels on 4h for breakout entries with 12h EMA trend filter and volume confirmation.
# Targets 20-35 trades/year on BTC/ETH by combining tight price-level breakouts with trend and volume filters.
# Works in bull markets via breakouts and in bear via mean-reversion at S1/R1 with trend filter avoiding counter-trend trades.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS_Refined"
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
    
    # Calculate Camarilla levels for each 4h bar using previous bar's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    for i in range(1, n):
        if np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1]):
            r1[i] = np.nan
            s1[i] = np.nan
        else:
            r1[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 12
            s1[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 12
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > vol_ma[i]  # Volume above average
        
        if position == 0:
            # Long: Break above R1 with uptrend and volume
            if close[i] > r1[i] and close[i] > ema_50_12h_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with downtrend and volume
            elif close[i] < s1[i] and close[i] < ema_50_12h_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close below S1 or trend reversal
            if close[i] < s1[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close above R1 or trend reversal
            if close[i] > r1[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals