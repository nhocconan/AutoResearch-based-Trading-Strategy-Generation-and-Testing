#!/usr/bin/env python3
name = "1d_WeeklyKeltnerBreakout"
timeframe = "1d"
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
    
    # 1w trend filter: 34 EMA on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Keltner Channel (20, 1.5)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 1.5 * atr
    kc_lower = ema_20 - 1.5 * atr
    
    # Volume filter: 20-period average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure Keltner and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period average
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price closes above upper Keltner in weekly uptrend, with volume spike
            if (close[i] > kc_upper[i] and 
                close[i] > ema_34_1w_aligned[i] and   # weekly uptrend filter
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price closes below lower Keltner in weekly downtrend, with volume spike
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema_34_1w_aligned[i] and   # weekly downtrend filter
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to middle of Keltner Channel (EMA20)
            if position == 1 and close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals