#!/usr/bin/env python3
"""
4h_Keltner_Channel_1dTrend_Volume
Hypothesis: Keltner Channel breakout with 1d EMA34 trend filter and volume confirmation.
Keltner Channel uses ATR-based bands to identify volatility breakouts.
In trending markets, price tends to stay outside channels; in ranging markets, it reverts to mean.
Trend filter ensures we only trade in direction of higher timeframe trend.
Volume confirmation filters weak breakouts. Works in both bull (breakouts above upper band) and bear (breakouts below lower band).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "4h_Keltner_Channel_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
    
    # ATR for Keltner Channel (20-period)
    atr_period = 20
    atr = np.full(n, np.nan)
    if n >= atr_period:
        tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
        tr = np.concatenate([[np.nan], tr])
        atr[atr_period-1] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # EMA20 for Keltner Channel middle line
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    # Keltner Channel bands
    keltner_mult = 2.0
    upper_band = ema20 + keltner_mult * atr
    lower_band = ema20 - keltner_mult * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        # Approximate 4h volume from 1d: 1d volume / 6 (since 24h/4h = 6)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Price breaks above upper Keltner band with uptrend and volume confirmation
            if close[i] > upper_band[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Keltner band with downtrend and volume confirmation
            elif close[i] < lower_band[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters Keltner Channel (below middle line) or trend reversal
            if close[i] < ema20[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Keltner Channel (above middle line) or trend reversal
            if close[i] > ema20[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals