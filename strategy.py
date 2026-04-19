# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Regime: 1d ADX < 20 = range (mean revert at Elder Ray extremes).
#          ADX > 25 = trend (follow Elder Ray zero-cross).
# Works in bull/bear by adapting to regime. Target: 50-150 trades over 4 years.

name = "6h_ElderRay_1dADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    # True Range
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr_ma = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_ma
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_ma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(ema13[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            if adx_val < 20:  # Range: mean revert at extremes
                if bull < -0.1 * close[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif bear > 0.1 * close[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
            else:  # Trend: follow momentum
                if bull > 0 and bear < 0:  # Strong bull
                    signals[i] = 0.25
                    position = 1
                elif bull < 0 and bear > 0:  # Strong bear
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: power fades or regime change
            if bull < 0 or (adx_val < 20 and bull < 0.05 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: power fades or regime change
            if bear > 0 or (adx_val < 20 and bear > -0.05 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals