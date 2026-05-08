#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Chop_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and chop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly ADX(14) for trend strength
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    tr1 = high_1w_series - low_1w_series
    tr2 = abs(high_1w_series - close_1w_series.shift(1))
    tr3 = abs(low_1w_series - close_1w_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean().values
    up_move = high_1w_series - high_1w_series.shift(1)
    down_move = low_1w_series.shift(1) - low_1w_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily Chop(14) for regime detection
    tr_daily = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr_daily[0] = high[0] - low[0]
    atr_14 = pd.Series(tr_daily).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14 + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend + strong trend + low chop (trending regime)
            if (close[i] > ema_34_1w_aligned[i] and 
                adx_1w_aligned[i] > 25 and
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + strong trend + low chop (trending regime)
            elif (close[i] < ema_34_1w_aligned[i] and 
                  adx_1w_aligned[i] > 25 and
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly downtrend OR chop increases (range regime)
            if (close[i] < ema_34_1w_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly uptrend OR chop increases (range regime)
            if (close[i] > ema_34_1w_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals