#!/usr/bin/env python3
"""
6h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX (12-period) combined with volume spikes and chop regime filtering works in both bull and bear markets.
Long: TRIX crosses above zero, volume spike, chop regime < 61.8 (trending)
Short: TRIX crosses below zero, volume spike, chop regime < 61.8 (trending)
Exit on opposite TRIX cross or chop regime > 61.8 (range). Uses 12h trend filter for higher timeframe bias.
Target: 12-37 trades/year per symbol.
"""

name = "6h_TRIX_VolumeSpike_Regime"
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
    
    # TRIX: 12-period EMA of EMA of EMA of close, then 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100
    trix_values = trix.fillna(0).values
    
    # Chop regime: using ATR(14) and highest high/lowest low over 14 periods
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i])
        lowest_low[i] = np.min(low[i-14:i])
    
    # Chop calculation: 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop = np.full(n, 50.0)  # default neutral
    for i in range(14, n):
        if highest_high[i] > lowest_low[i]:
            sum_atr = np.sum(tr[i-14:i])
            chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((highest_high[i] - lowest_low[i]) + 1e-10)
        else:
            chop[i] = 50.0
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    # 12h trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    ema_30_12h = pd.Series(df_12h['close']).ewm(span=30, adjust=False, min_periods=30).mean().values
    uptrend_12h = df_12h['close'].values > ema_30_12h
    downtrend_12h = df_12h['close'].values < ema_30_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        trix_now = trix_values[i]
        trix_prev = trix_values[i-1]
        vol_conf = volume_conf[i]
        chop_now = chop[i]
        uptrend_htf = uptrend_12h_aligned[i]
        downtrend_htf = downtrend_12h_aligned[i]
        
        if position == 0:
            # LONG: TRIX crosses above zero, volume chop < 61.8 (trending), volume confirmation
            if trix_prev <= 0 and trix_now > 0 and chop_now < 61.8 and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, volume chop < 61.8 (trending), volume confirmation
            elif trix_prev >= 0 and trix_now < 0 and chop_now < 61.8 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR chop > 61.8 (range)
            if trix_prev >= 0 and trix_now < 0 or chop_now > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR chop > 61.8 (range)
            if trix_prev <= 0 and trix_now > 0 or chop_now > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals