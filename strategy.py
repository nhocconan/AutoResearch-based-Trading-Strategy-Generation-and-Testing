#!/usr/bin/env python3
name = "12h_Pivots_Breakout_Trend"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate daily ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Pivot levels: R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volatility filter using daily ATR ratio to avoid chop
    high_low_12h = high - low
    high_close_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(high_low_12h, np.maximum(high_close_12h, low_close_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for volume MA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range filter: avoid trading in low volatility chop
        if atr_12h[i] < 0.5 * atr_1d_aligned[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 1.5
            uptrend = close[i] > ema_34_aligned[i]
            
            if close[i] > r1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and in downtrend
            elif close[i] < s1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or volatility spike
            if close[i] < pivot_aligned[i] or atr_12h[i] > 3 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or volatility spike
            if close[i] > pivot_aligned[i] or atr_12h[i] > 3 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Pivot Breakout with daily trend and volume confirmation.
# Uses daily pivot points (R1/S1) as breakout levels, daily EMA(34) for trend filter,
# and volume confirmation to avoid false breaks. Volatility filter prevents whipsaws
# in choppy markets. Works in bull (buy R1 breaks in uptrend) and bear (sell S1 breaks
# in downtrend). Position size 0.25 balances risk and keeps trade frequency ~10-25/year.