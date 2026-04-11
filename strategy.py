#!/usr/bin/env python3
# 1h_4d_1d_adx_volume_v1
# Strategy: 1h ADX trend strength with volume confirmation and 4h/1d trend filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: ADX > 25 indicates strong trend. In bull markets: long when ADX rising, price above 4h/1d EMA50, volume > 1.5x average. In bear markets: short when ADX rising, price below 4h/1d EMA50, volume > 1.5x average. Uses 4h and 1d EMA50 for trend alignment to avoid counter-trend trades. Low frequency (~15-35/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_adx_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(val, period):
        result = np.zeros_like(val)
        result[period-1] = np.nansum(val[:period])
        for i in range(period, len(val)):
            result[i] = result[i-1] - (result[i-1] / period) + val[i]
        return result
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth(dx, 14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend alignment: price above/both EMAs for long, below/both for short
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # ADX rising and strong trend
        adx_rising = adx[i] > adx[i-1]
        strong_trend = adx[i] > 25
        
        # Entry logic: ADX strength + volume + trend alignment
        if (strong_trend and adx_rising and (plus_di[i] > minus_di[i]) and 
            uptrend_4h and uptrend_1d and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.20
        elif (strong_trend and adx_rising and (minus_di[i] > plus_di[i]) and 
              downtrend_4h and downtrend_1d and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: ADX weakening or trend change
        elif position == 1 and (adx[i] < 20 or not (uptrend_4h and uptrend_1d)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx[i] < 20 or not (downtrend_4h and downtrend_1d)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals