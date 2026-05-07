#!/usr/bin/env python3
name = "6h_Keltner_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily ATR(20) for Keltner channels
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20_1d = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily EMA(20) for trend
    ema_20_1d = df_1d['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 6h
    atr_20_6h = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Keltner Channels on 6h using daily ATR (lookback period=20)
    ema_20_6h_series = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20_6h_series + (2.0 * atr_20_6h)
    lower_keltner = ema_20_6h_series - (2.0 * atr_20_6h)
    
    # Volume filter: volume > 1.5 * 20-period average volume on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close breaks above upper Keltner + daily uptrend + volume filter
            if close[i] > upper_keltner[i] and close[i] > ema_20_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below lower Keltner + daily downtrend + volume filter
            elif close[i] < lower_keltner[i] and close[i] < ema_20_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close below EMA(20) or opposite Keltner break
            if close[i] < ema_20_6h[i] or close[i] < lower_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close above EMA(20) or opposite Keltner break
            if close[i] > ema_20_6h[i] or close[i] > upper_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals