#!/usr/bin/env python3
name = "1h_Keltner_Reversal_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: 21 EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1h Keltner Channel (20, 2.0)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # Volume filter: 20-period average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure Keltner and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(atr[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price closes below lower Keltner (oversold) in 4h uptrend, with volume spike
            if (close[i] < kc_lower[i] and 
                close[i] > ema_21_4h_aligned[i] and   # 4h uptrend filter
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: Price closes above upper Keltner (overbought) in 4h downtrend, with volume spike
            elif (close[i] > kc_upper[i] and 
                  close[i] < ema_21_4h_aligned[i] and   # 4h downtrend filter
                  volume_spike):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: Price returns to middle of Keltner Channel (EMA20)
            if position == 1 and close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals