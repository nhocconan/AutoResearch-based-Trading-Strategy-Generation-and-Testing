#!/usr/bin/env python3
name = "6h_ADX_20_Trend_with_ElderRay_1d_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX(14) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2 = (high_series - close_series.shift()).abs()
    tr3 = (low_series - close_series.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    up_move = high_series.diff()
    down_move = low_series.shift() - low_series
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(up_move)
    minus_dm = pd.Series(down_move)
    
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_strong = adx_values > 20
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d trend filter: price above/below EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Enough for ADX and EMA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(adx_values[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(trend_up_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Strong ADX + Bull Power positive + 1d uptrend
            if adx_strong[i] and bull_power[i] > 0 and trend_up_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong ADX + Bear Power negative + 1d downtrend
            elif adx_strong[i] and bear_power[i] < 0 and not trend_up_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Weak ADX OR Bear Power negative (momentum shift)
            if not adx_strong[i] or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Weak ADX OR Bull Power positive (momentum shift)
            if not adx_strong[i] or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals