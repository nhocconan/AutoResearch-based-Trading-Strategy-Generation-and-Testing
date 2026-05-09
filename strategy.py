#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_AdaptiveKeltnerBreakout_1dTrend_Volume"
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
    
    # Get 1d data for trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1d data for ATR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA(20) on 1d for trend
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner Channel on 6h using 1d ATR
    # Upper = EMA(20) + 2 * ATR(14)
    # Lower = EMA(20) - 2 * ATR(14)
    keltner_upper = ema20_1d + 2 * atr_1d
    keltner_lower = ema20_1d - 2 * atr_1d
    
    # Align all to 6h
    ema20_1d_6h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    keltner_upper_6h = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_6h = align_htf_to_ltf(prices, df_1d, keltner_lower)
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1d_6h[i]) or np.isnan(keltner_upper_6h[i]) or 
            np.isnan(keltner_lower_6h[i]) or np.isnan(atr_1d_6h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema20_1d_6h[i]
        upper = keltner_upper_6h[i]
        lower = keltner_lower_6h[i]
        atr_val = atr_1d_6h[i]
        vol_ok = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: break above upper band with volume and uptrend
            if close[i] > upper and vol_ok and close[i] > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume and downtrend
            elif close[i] < lower and vol_ok and close[i] < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA or opposite band touch
            if close[i] < ema_trend or close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above EMA or opposite band touch
            if close[i] > ema_trend or close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals