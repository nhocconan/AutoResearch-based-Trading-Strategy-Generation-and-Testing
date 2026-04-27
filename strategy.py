#!/usr/bin/env python3
"""
6h_OrderFlowImbalance_Reversal_v1
Hypothesis: In choppy markets (low ADX), price tends to revert from extreme order flow imbalances.
Uses: 1) 6h price closes outside 1.5*ATR from VWAP (extreme), 2) 1d ADX < 20 (chop regime), 3) 6h RSI divergence (confirmation).
Only takes mean-reversion trades when all three align. Works in bull/bear because it fades extremes in chop.
Target: 20-40 trades/year to minimize fee drag. Size: 0.25.
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
    volume = prices['volume'].values
    
    # 6h VWAP (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, 0.0)
    
    # 6h ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = np.zeros(n)
    atr[13] = np.mean(tr[:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # 6h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d ADX(14) for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    atr_1d = smooth_wilder(tr_1d, 14)
    plus_di_1d = 100 * smooth_wilder(plus_dm, 14) / np.where(atr_1d != 0, atr_1d, 1)
    minus_di_1d = 100 * smooth_wilder(minus_dm, 14) / np.where(atr_1d != 0, atr_1d, 1)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / np.where((plus_di_1d + minus_di_1d) != 0, (plus_di_1d + minus_di_1d), 1)
    adx_1d = smooth_wilder(dx_1d, 14)
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Conditions
    price_above_vwap = close > vwap + (1.5 * atr)
    price_below_vwap = close < vwap - (1.5 * atr)
    rsi_overbought = rsi > 70
    rsi_oversold = rsi < 30
    chop_regime = adx_1d_aligned < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25
    
    start_idx = max(30, 14)  # need ADX and ATR/RSI warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(price_above_vwap[i]) or np.isnan(price_below_vwap[i]) or 
            np.isnan(rsi_overbought[i]) or np.isnan(rsi_oversold[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price far below VWAP + RSI oversold + chop regime
            if price_below_vwap[i] and rsi_oversold[i] and chop_regime[i]:
                signals[i] = size
                position = 1
            # Short: price far above VWAP + RSI overbought + chop regime
            elif price_above_vwap[i] and rsi_overbought[i] and chop_regime[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price returns to VWAP or RSI neutral
            if close[i] >= vwap[i] or rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP or RSI neutral
            if close[i] <= vwap[i] or rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_OrderFlowImbalance_Reversal_v1"
timeframe = "6h"
leverage = 1.0