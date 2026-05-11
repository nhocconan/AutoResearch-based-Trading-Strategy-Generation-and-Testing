#!/usr/bin/env python3
name = "6h_Supertrend_ADX_Pullback"
timeframe = "6h"
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
    
    # Load 1w and 1d data ONCE
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1w Supertrend for long-term trend (ATR=10, mult=3)
    atr_period = 10
    mult = 3
    tr1 = np.maximum(df_1w['high'], np.roll(df_1w['close'], 1)) - np.minimum(df_1w['low'], np.roll(df_1w['close'], 1))
    tr2 = np.abs(np.roll(df_1w['close'], 1) - df_1w['high'])
    tr3 = np.abs(np.roll(df_1w['close'], 1) - df_1w['low'])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    up = ((df_1w['high'] + df_1w['low']) / 2) - (mult * atr)
    down = ((df_1w['high'] + df_1w['low']) / 2) + (mult * atr)
    up = np.where(np.isnan(up), 0, up)
    down = np.where(np.isnan(down), 0, down)
    st = np.zeros_like(df_1w['close'])
    dir = np.ones_like(df_1w['close'])  # 1 for uptrend, -1 for downtrend
    for i in range(1, len(df_1w)):
        if df_1w['close'][i-1] > st[i-1]:
            st[i] = max(up[i], st[i-1])
        else:
            st[i] = min(down[i], st[i-1])
        if df_1w['close'][i] > st[i]:
            dir[i] = 1
        else:
            dir[i] = -1
    st_1w = st
    dir_1w = dir
    
    # 1d ADX for trend strength
    period = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d, np.roll(close_1d, 1)) - np.minimum(low_1d, np.roll(close_1d, 1))
    tr2 = np.abs(np.roll(close_1d, 1) - high_1d)
    tr3 = np.abs(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    dx = np.zeros_like(close_1d)
    dmplus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dmminus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dmplus[0] = 0
    dmminus[0] = 0
    tr_sum = np.zeros_like(tr)
    dmplus_sum = np.zeros_like(dmplus)
    dmminus_sum = np.zeros_like(dmminus)
    for i in range(len(tr)):
        if i == 0:
            tr_sum[i] = tr[i]
            dmplus_sum[i] = dmplus[i]
            dmminus_sum[i] = dmminus[i]
        else:
            tr_sum[i] = tr_sum[i-1] + tr[i] - tr_sum[i-1] / period
            dmplus_sum[i] = dmplus_sum[i-1] + dmplus[i] - dmplus_sum[i-1] / period
            dmminus_sum[i] = dmminus_sum[i-1] + dmminus[i] - dmminus_sum[i-1] / period
    plus_di = 100 * dmplus_sum / tr_sum
    minus_di = 100 * dmminus_sum / tr_sum
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < period:
            adx[i] = 0
        else:
            if i == period:
                adx[i] = np.mean(dx[:period])
            else:
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    adx_1d = adx
    
    # Align HTF indicators to 6h
    st_1w_aligned = align_htf_to_ltf(prices, df_1w, st_1w)
    dir_1w_aligned = align_htf_to_ltf(prices, df_1w, dir_1w)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 60-period EMA for pullback entry
    ema60 = pd.Series(close).ewm(span=60, min_periods=60, adjust=False).mean().values
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(st_1w_aligned[i]) or np.isnan(dir_1w_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema60[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        uptrend_1w = dir_1w_aligned[i] == 1
        downtrend_1w = dir_1w_aligned[i] == -1
        strong_trend = adx_1d_aligned[i] > 25
        pullback_long = close[i] < ema60[i] and close[i] > st_1w_aligned[i]
        pullback_short = close[i] > ema60[i] and close[i] < st_1w_aligned[i]
        
        if position == 0:
            # Long: 1w uptrend + strong trend + pullback to EMA60 above Supertrend
            if uptrend_1w and strong_trend and pullback_long and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: 1w downtrend + strong trend + pullback to EMA60 below Supertrend
            elif downtrend_1w and strong_trend and pullback_short and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: trend weakens or price crosses Supertrend
            if position == 1:
                if adx_1d_aligned[i] < 20 or close[i] < st_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if adx_1d_aligned[i] < 20 or close[i] > st_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals