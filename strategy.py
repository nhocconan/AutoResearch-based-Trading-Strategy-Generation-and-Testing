#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Keltner_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Keltner
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for volume context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily ATR(10) for Keltner channels
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_6h = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Keltner channels: EMA20 ± 2*ATR
    upper = ema_20_1d + 2 * atr_10
    lower = ema_20_1d - 2 * atr_10
    upper_6h = align_htf_to_ltf(prices, df_1d, upper)
    lower_6h = align_htf_to_ltf(prices, df_1d, lower)
    
    # 12h volume average for spike detection
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_6h = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_6h[i]) or np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or
            np.isnan(vol_ma_12h_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma_12h_6h[i]  # Volume confirmation
        
        # Session filter: trade only during active hours (UTC 8-20)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: close above upper Keltner, uptrend (close > EMA20), volume spike
            if (close[i] > upper_6h[i] and 
                close[i] > ema_20_6h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: close below lower Keltner, downtrend (close < EMA20), volume spike
            elif (close[i] < lower_6h[i] and 
                  close[i] < ema_20_6h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA20 (trend reversal) or below lower Keltner
            if close[i] < ema_20_6h[i] or close[i] < lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above EMA20 (trend reversal) or above upper Keltner
            if close[i] > ema_20_6h[i] or close[i] > upper_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals