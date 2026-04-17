#!/usr/bin/env python3
"""
Hypothesis: 1d Exponential Moving Average (EMA) crossover with 1w volume confirmation and 1d ADX trend strength filter.
Uses EMA(21) and EMA(50) crossovers with weekly volume > 1.5x 50-period average and ADX(14) > 20 to filter trades.
Targets 15-25 trades/year to avoid fee drag. Works in bull (trend-following crossovers) and bear (ADX filter avoids chop).
"""

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
    
    # === Daily EMA(21) and EMA(50) for crossover signals ===
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Weekly EMA(50) for trend direction (optional filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Weekly volume confirmation ===
    volume_1w = df_1w['volume'].values
    vol_ma_50 = pd.Series(volume_1w).rolling(window=50, min_periods=50).mean().values
    vol_ma_50_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_50)
    
    # === Daily ADX(14) for trend strength filter ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_dm_smoothed = np.zeros_like(plus_dm)
    minus_dm_smoothed = np.zeros_like(minus_dm)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])  # 14-period ATR
    plus_dm_smoothed[13] = np.mean(plus_dm[1:14])
    minus_dm_smoothed[13] = np.mean(minus_dm[1:14])
    
    # Wilder's smoothing
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.where(atr != 0, (plus_dm_smoothed / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_dm_smoothed / atr) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros_like(dx)
    # Initial ADX value (average of first 14 DX values)
    adx[27] = np.mean(dx[14:28]) if n > 28 else 0
    # Wilder's smoothing for ADX
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_50_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume spike: current 1w volume > 1.5x 50-period average
        df_1w_current = get_htf_data(prices, '1w')
        vol_1w_current = df_1w_current['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w_current, vol_1w_current)
        vol_spike = vol_1w_aligned[i] > vol_ma_50_aligned[i] * 1.5
        
        # Trend filter: ADX > 20 indicates strong trend
        strong_trend = adx[i] > 20
        
        # EMA crossover signals
        ema_bullish = ema_21[i] > ema_50[i]  # Fast above slow = bullish
        ema_bearish = ema_21[i] < ema_50[i]  # Fast below slow = bearish
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: EMA bullish crossover + volume spike + strong trend
            if ema_bullish and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: EMA bearish crossover + volume spike + strong trend
            elif ema_bearish and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite crossover
        elif position == 1:
            # Exit long if EMA turns bearish
            if ema_bearish:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if EMA turns bullish
            if ema_bullish:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA21_50_1wVolume1.5x_ADX20"
timeframe = "1d"
leverage = 1.0