#!/usr/bin/env python3
"""
6h_ADX_BollingerBand_Reversion_1dTrend
Hypothesis: Mean reversion at Bollinger Bands (±2 std) in strong trends (ADX>25) with 1-day EMA50 trend filter. 
Works in bull/bear by only trading mean-reversion pullbacks in established trends, avoiding chop.
Target: 60-120 trades over 4 years (15-30/year). Uses discrete position sizing (0.25) to minimize fee drag.
"""

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
    
    # Calculate 6h indicators
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_ma = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    for i in range(bb_period, n):
        bb_ma[i] = np.mean(close[i-bb_period:i])
        bb_std[i] = np.std(close[i-bb_period:i])
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # ADX (14) - need +DI, -DI, TR
    adx_period = 14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr_ma = np.full(n, np.nan)
    plus_di_ma = np.full(n, np.nan)
    minus_di_ma = np.full(n, np.nan)
    for i in range(adx_period, n):
        if i == adx_period:
            tr_ma[i] = np.sum(tr[0:adx_period+1])
            plus_di_ma[i] = np.sum(plus_dm[0:adx_period+1])
            minus_di_ma[i] = np.sum(minus_dm[0:adx_period+1])
        else:
            tr_ma[i] = tr_ma[i-1] - (tr_ma[i-1] / adx_period) + tr[i]
            plus_di_ma[i] = plus_di_ma[i-1] - (plus_di_ma[i-1] / adx_period) + plus_dm[i]
            minus_di_ma[i] = minus_di_ma[i-1] - (minus_di_ma[i-1] / adx_period) + minus_dm[i]
    
    plus_di = np.where(tr_ma != 0, 100 * plus_di_ma / tr_ma, 0)
    minus_di = np.where(tr_ma != 0, 100 * minus_di_ma / tr_ma, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full(n, np.nan)
    for i in range(adx_period, n):
        if i == adx_period:
            adx[i] = np.mean(dx[0:adx_period+1])
        else:
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Calculate 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema50_1d[i] = np.mean(close_1d[0:51])
        else:
            k = 2 / (50 + 1)
            ema50_1d[i] = close_1d[i] * k + ema50_1d[i-1] * (1 - k)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, adx_period, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(adx[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: pullback to lower BB in uptrend (ADX>25, price>EMA50_1d)
            if (close[i] <= bb_lower[i] and adx[i] > 25 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: pullback to upper BB in downtrend (ADX>25, price<EMA50_1d)
            elif (close[i] >= bb_upper[i] and adx[i] > 25 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to mean (middle BB) or trend weakness
            if (close[i] >= bb_ma[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to mean (middle BB) or trend weakness
            if (close[i] <= bb_ma[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_BollingerBand_Reversion_1dTrend"
timeframe = "6h"
leverage = 1.0