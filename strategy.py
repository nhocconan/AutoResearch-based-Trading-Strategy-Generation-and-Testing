#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Keltner_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(20) on 1d for Keltner channel width
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate 20-period EMA on 1d close for trend filter and Keltner center
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Keltner bands: EMA(20) ± 2 * ATR(20)
    upper_keltner = ema_20_1d + 2 * atr_20_1d
    lower_keltner = ema_20_1d - 2 * atr_20_1d
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_keltner_val = upper_keltner_aligned[i]
        lower_keltner_val = lower_keltner_aligned[i]
        ema_1d = ema_20_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Close > Upper Keltner AND price > 1d EMA20 (uptrend) AND volume > 2.5x average
            if close[i] > upper_keltner_val and close[i] > ema_1d and vol > 2.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < Lower Keltner AND price < 1d EMA20 (downtrend) AND volume > 2.5x average
            elif close[i] < lower_keltner_val and close[i] < ema_1d and vol > 2.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < EMA(20) OR trend reverses (price < 1d EMA20)
            if close[i] < ema_1d or close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > EMA(20) OR trend reverses (price > 1d EMA20)
            if close[i] > ema_1d or close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals