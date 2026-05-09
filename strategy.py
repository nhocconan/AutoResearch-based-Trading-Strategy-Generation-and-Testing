#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Keltner_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for ATR (used in Keltner channels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period ATR on 1d high/low/close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate 20-period EMA on 12h close for Keltner middle line
    ema_20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema_20_12h + 2.0 * atr_20_1d_aligned
    keltner_lower = ema_20_12h - 2.0 * atr_20_1d_aligned
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_50_1w_aligned[i]
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Close > Upper Keltner AND price > 1w EMA50 (uptrend) AND volume > 2.0x average
            if close[i] > upper and close[i] > ema_1w and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < Lower Keltner AND price < 1w EMA50 (downtrend) AND volume > 2.0x average
            elif close[i] < lower and close[i] < ema_1w and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < Lower Keltner OR trend reverses (price < 1w EMA50)
            if close[i] < lower or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > Upper Keltner OR trend reverses (price > 1w EMA50)
            if close[i] > upper or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals