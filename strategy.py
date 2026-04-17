#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high/low for pivot levels (using daily data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Support 1: S1 = 2*P - H
    s1_1d = 2 * pivot_1d - high_1d
    # Resistance 1: R1 = 2*P - L
    r1_1d = 2 * pivot_1d - low_1d
    
    # Align pivot levels to 4h timeframe (with 1-day delay for completion)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h ADX for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(adx[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        atr_val = atr_1d_aligned[i]
        
        if position == 0:
            # Long when price crosses above R1 with volume and trend confirmation
            if price > r1_aligned[i] and vol > 1.5 * vol_ma and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short when price crosses below S1 with volume and trend confirmation
            elif price < s1_aligned[i] and vol > 1.5 * vol_ma and adx[i] > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to pivot or trend weakens
            if price < pivot_aligned[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to pivot or trend weakens
            if price > pivot_aligned[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_R1S1_Volume_ADX"
timeframe = "4h"
leverage = 1.0