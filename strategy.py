#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Keltner_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Keltner Channel parameters (20, 2.0) on 6h data
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean()
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above upper Keltner + above 1d EMA34 + volume spike
            if (close[i] > upper_keltner[i] and 
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close below lower Keltner + below 1d EMA34 + volume spike
            elif (close[i] < lower_keltner[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below EMA20 (middle of Keltner) OR below 1d EMA34
            if close[i] < ema_20[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above EMA20 OR above 1d EMA34
            if close[i] > ema_20[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals