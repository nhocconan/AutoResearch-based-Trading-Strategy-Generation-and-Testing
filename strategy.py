#!/usr/bin/env python3
"""
6h_ADX_Trend_With_VolumeSpike
Hypothesis: Use ADX(14) to identify strong trends (ADX>25) and enter long/short in the direction of the 1d EMA50 when price breaks the ATR-based band, confirmed by volume spike. Exit when trend weakens (ADX<20). Designed for 6h timeframe to capture trends in both bull and bear markets with moderate trade frequency.
"""

name = "6h_ADX_Trend_With_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility and band width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ADX(14)
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed +DM, -DM, TR
    tr_14 = np.full(n, np.nan)
    plus_dm_14 = np.full(n, np.nan)
    minus_dm_14 = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            tr_14[i] = np.sum(tr[1:15])
            plus_dm_14[i] = np.sum(plus_dm[1:15])
            minus_dm_14[i] = np.sum(minus_dm[1:15])
        else:
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
            plus_dm_14[i] = plus_dm_14[i-1] - (plus_dm_14[i-1] / 14) + plus_dm[i]
            minus_dm_14[i] = minus_dm_14[i-1] - (minus_dm_14[i-1] / 14) + minus_dm[i]
    
    # DI+ and DI-
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(14, n):
        if tr_14[i] > 0:
            plus_di[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di[i] = (minus_dm_14[i] / tr_14[i]) * 100
    
    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(14, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    adx = np.full(n, np.nan)
    for i in range(28, n):  # ADX needs 2*period
        if i == 28:
            adx[i] = np.mean(dx[15:29])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Calculate 1d EMA50 for trend filter (using HTF data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50, 20)  # Ensure ADX, EMA, and volume SMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(vol_sma[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        # Weak trend filter: ADX < 20 to exit
        weak_trend = adx[i] < 20
        
        # ATR-based bands: entry when price breaks 1.5 * ATR from close
        upper_band = close[i-1] + 1.5 * atr[i]
        lower_band = close[i-1] - 1.5 * atr[i]
        
        # Volume confirmation: current volume > 2.0 * average volume
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Enter long: strong trend, price breaks upper band, above 1d EMA50, volume confirmation
            if strong_trend and close[i] > upper_band and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: strong trend, price breaks lower band, below 1d EMA50, volume confirmation
            elif strong_trend and close[i] < lower_band and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakens or price crosses below 1d EMA50
            if weak_trend or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakens or price crosses above 1d EMA50
            if weak_trend or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals