#!/usr/bin/env python3
"""
4h_WilliamsVixFix_VolumeSpike_v1
Williams Vix Fix (WVF) > 0.8 + Volume spike > 1.5x avg volume + RSI(14) > 50 for long, < 50 for short.
Uses 1d ADX > 25 for trend filter (avoid chop). Exit when WVF < 0.3 or ADX < 20.
Designed to capture volatility expansion moves in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Williams Vix Fix (WVF) ===
    # WVF = ((Highest Close in n-period - Low) / Highest Close in n-period) * 100
    # We use 22-period as per Larry Williams
    highest_close = pd.Series(close).rolling(window=22, min_periods=22).max()
    wvf = ((highest_close - low) / highest_close) * 100
    wvf = wvf.values  # already 0-100 scale
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume spike (1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / (vol_ma + 1e-10)  # ratio
    
    # === 1d ADX(14) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(wvf[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_spike[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: WVF > 0.8 (high fear), volume spike, RSI > 50, ADX > 25
            if (wvf[i] > 80 and 
                vol_spike[i] > 1.5 and 
                rsi[i] > 50 and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
                continue
            # Short: WVF > 0.8 (high fear), volume spike, RSI < 50, ADX > 25
            elif (wvf[i] > 80 and 
                  vol_spike[i] > 1.5 and 
                  rsi[i] < 50 and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: WVF < 0.3 (low fear) OR ADX < 20
            if (wvf[i] < 30 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WVF < 0.3 (low fear) OR ADX < 20
            if (wvf[i] < 30 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsVixFix_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0