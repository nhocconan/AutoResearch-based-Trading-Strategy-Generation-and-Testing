#!/usr/bin/env python3
# 6h_ADX_200EMA_Trend_With_Pullback
# Hypothesis: Strong trends (ADX>25) above/below 200 EMA provide directional bias. 
# Enter on pullbacks to 50 EMA with volume confirmation. 
# Works in bull markets via pullback longs in uptrends and in bear via pullback shorts in downtrends.
# Low trade frequency (target: 20-40/year) to minimize fee drag.

name = "6h_ADX_200EMA_Trend_With_Pullback"
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
    
    # Daily trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Weekly ADX for trend strength
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(np.roll(low_1w, 1), prepend=low_1w[0])  # low[t] - low[t-1]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (Wilder smoothing = EMA with alpha=1/period)
    def wilders_smooth(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            res[period-1] = np.mean(arr[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 6h EMA50 for pullback entries
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (20-period average on 6h = ~5 days)
    def mean_arr(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period-1, len(arr)):
                res[i] = np.mean(arr[i-period+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 200) + 20  # need enough history
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: uptrend (price > EMA200, ADX>25) and pullback to EMA50 with volume
            if close[i] > ema_200_aligned[i] and adx_aligned[i] > 25 and \
               low[i] <= ema_50[i] * 1.005 and high[i] >= ema_50[i] * 0.995 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price < EMA200, ADX>25) and pullback to EMA50 with volume
            elif close[i] < ema_200_aligned[i] and adx_aligned[i] > 25 and \
                 high[i] >= ema_50[i] * 0.995 and low[i] <= ema_50[i] * 1.005 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks (price < EMA200 or ADX < 20) or strong reversal
            if close[i] < ema_200_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks (price > EMA200 or ADX < 20) or strong reversal
            if close[i] > ema_200_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals