#!/usr/bin/env python3
"""
4h_Keltner_Channel_1dTrend_Volume
Hypothesis: Keltner breakouts on 4h timeframe with 1d EMA34 trend filter and volume confirmation.
In trending markets, price tends to break and continue in direction of trend.
Works in both bull (breakouts above upper band) and bear (breakdowns below lower band).
Target 20-50 trades per year to minimize fee drag.
"""

name = "4h_Keltner_Channel_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtt_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h ATR(10) for Keltner channels
    atr_period = 10
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # 4h EMA20 for middle band
    ema_period = 20
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period-1] = np.mean(close[:ema_period])
        alpha = 2 / (ema_period + 1)
        for i in range(ema_period, n):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    
    # Keltner bands
    keltner_mult = 2.0
    upper = ema + keltner_mult * atr
    lower = ema - keltner_mult * atr
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, ema_period, 34, 1)
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0  # 6x 4h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Break above upper Keltner band in uptrend with volume confirmation
            if close[i] > upper[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Keltner band in downtrend with volume confirmation
            elif close[i] < lower[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below EMA middle band or trend reversal
            if close[i] < ema[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above EMA middle band or trend reversal
            if close[i] > ema[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals