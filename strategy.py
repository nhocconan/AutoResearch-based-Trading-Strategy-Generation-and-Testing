#!/usr/bin/env python3
name = "12h_TRIX_Volume_Spike_Regime_1wTrend"
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
    
    # ===== 1W Trend Filter (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w TRIX (12-period)
    ema1 = pd.Series(close_1w).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix_raw = 100 * (ema3.diff() / ema3.shift(1))
    trix = trix_raw.fillna(0).values
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # ===== TRIX Signal (LTF) =====
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix_ltf = 100 * (ema3.diff() / ema3.shift(1))
    trix_ltf = trix_ltf.fillna(0).values
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    # ===== Choppy Market Filter (using 1d CHOP) =====
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ADX(14) components
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Chop = (log10(sum(TR)/ATR) / log10(period)) * 100
    chop = (np.log10(pd.Series(tr).rolling(window=14, min_periods=14).sum().values / (atr + 1e-10)) / np.log10(14)) * 100
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_ltf[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + chop < 61.8 (trending) + volume spike + 1w TRIX positive
            if (trix_ltf[i] > 0 and trix_ltf[i-1] <= 0 and 
                chop_aligned[i] < 61.8 and 
                vol_spike[i] and 
                trix_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + chop < 61.8 (trending) + volume spike + 1w TRIX negative
            elif (trix_ltf[i] < 0 and trix_ltf[i-1] >= 0 and 
                  chop_aligned[i] < 61.8 and 
                  vol_spike[i] and 
                  trix_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop > 61.8 (choppy)
            if trix_ltf[i] < 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop > 61.8 (choppy)
            if trix_ltf[i] > 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals