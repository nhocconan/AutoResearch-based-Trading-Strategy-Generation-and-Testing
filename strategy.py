# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from math import log10
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_200Day_EMA_Bull_Bear_Momentum_With_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 200EMA and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d 200EMA
    close_1d = pd.Series(df_1d['close'].values)
    ema_200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1d volume average (20-day)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation (20-period MA on 6h)
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # 6h price momentum (rate of change over 6 periods = 1 day)
    close_series = pd.Series(close)
    roc_6 = ((close_series / close_series.shift(6)) - 1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_current[i]) or np.isnan(roc_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        price_above_200ema = close[i] > ema_200_1d_aligned[i]
        price_below_200ema = close[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: Above 200EMA, positive momentum, volume
            if price_above_200ema and roc_6[i] > 0.01 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Below 200EMA, negative momentum, volume
            elif price_below_200ema and roc_6[i] < -0.01 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Falls below 200EMA or momentum turns negative
            if close[i] < ema_200_1d_aligned[i] or roc_6[i] < -0.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Rises above 200EMA or momentum turns positive
            if close[i] > ema_200_1d_aligned[i] or roc_6[i] > 0.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals