#!/usr/bin/env python3
"""
4h_ADX_Trend_RSI_MeanReversion
Hypothesis: In strong trends (ADX>25), RSI extremes provide high-probability mean-reversion entries. 
Combines ADX trend filter with RSI mean-reversion to work in both bull and bear markets.
Target: 20-40 trades/year on 4h timeframe with strict entry conditions.
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
    
    # Calculate 1-day ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR for ADX calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR (14-period)
    atr = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed DM and ATR
    plus_dm_smooth = np.full(len(close_1d), np.nan)
    minus_dm_smooth = np.full(len(close_1d), np.nan)
    atr_smooth = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            plus_dm_smooth[i] = np.nansum(plus_dm[1:15])
            minus_dm_smooth[i] = np.nansum(minus_dm[1:15])
            atr_smooth[i] = atr[i]
        else:
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            atr_smooth[i] = atr_smooth[i-1] - (atr_smooth[i-1] / 14) + atr[i]
    
    # DI and ADX
    plus_di = np.full(len(close_1d), np.nan)
    minus_di = np.full(len(close_1d), np.nan)
    dx = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX (smoothed DX)
    adx = np.full(len(close_1d), np.nan)
    for i in range(28, len(close_1d)):  # 14 + 14 for smoothing
        if i == 28:
            adx[i] = np.nanmean(dx[15:29])
        else:
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # RSI (14-period) on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 28  # ADX needs 28 periods
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Strong trend filter: ADX > 25
            if adx_aligned[i] > 25:
                # Mean reversion in strong trend: RSI < 30 for long, RSI > 70 for short
                if rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend weakens
            if rsi[i] >= 50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend weakens
            if rsi[i] <= 50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_Trend_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0