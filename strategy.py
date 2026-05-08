#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX + Directional Movement with 12-hour Supertrend filter and volume confirmation
# Long when ADX > 25 + +DI > -DI + price > 12h Supertrend + volume spike
# Short when ADX > 25 + -DI > +DI + price < 12h Supertrend + volume spike
# ADX filters weak trends, reducing whipsaw in ranging markets
# Supertrend provides dynamic support/resistance from higher timeframe
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_ADX_Supertrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data once for Supertrend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range and ATR for 12h
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align with close_12h
    
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_12h = hl2_12h + (3.0 * atr_12h)
    lower_12h = hl2_12h - (3.0 * atr_12h)
    
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.full_like(close_12h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend_12h[i-1]):
            supertrend_12h[i] = lower_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i] > supertrend_12h[i-1]:
                supertrend_12h[i] = max(lower_12h[i], supertrend_12h[i-1])
                direction_12h[i] = 1
            else:
                supertrend_12h[i] = min(upper_12h[i], supertrend_12h[i-1])
                direction_12h[i] = -1
    
    supertrend_12h_val = supertrend_12h * direction_12h  # positive for uptrend, negative for downtrend
    supertrend_12h_signal = direction_12h  # 1 = uptrend, -1 = downtrend
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h_signal)
    
    # Calculate ADX, +DI, -DI (14-period)
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_6h
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_6h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(supertrend_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        supertrend_dir = supertrend_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: ADX > 25 + +DI > -DI + 12h uptrend + volume spike
            if adx_val > 25 and plus_di_val > minus_di_val and supertrend_dir > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 + -DI > +DI + 12h downtrend + volume spike
            elif adx_val > 25 and minus_di_val > plus_di_val and supertrend_dir < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 OR trend reversal
            if adx_val < 20 or plus_di_val < minus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 OR trend reversal
            if adx_val < 20 or minus_di_val < plus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals