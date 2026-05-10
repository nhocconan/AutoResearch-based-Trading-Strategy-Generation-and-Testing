#!/usr/bin/env python3
# 6h_KAMA_Trend_With_Volume_Spike_and_CHOP_Filter
# Hypothesis: KAMA adapts to market efficiency, filtering noise in choppy markets and capturing true trends.
# Combined with volume spike (institutional participation) and CHOP > 61.8 (range regime) for mean reversion at extremes.
# Works in bull/bear: KAMA follows trend, volume confirms strength, CHOP regime avoids false signals.
# Target: 50-150 total trades over 4 years (~12-37/year) on 6h timeframe.

name = "6h_KAMA_Trend_With_Volume_Spike_and_CHOP_Filter"
timeframe = "6h"
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
    
    # Get daily data for trend filter and CHOP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA (6h) - adaptive moving average
    # Efficiency Ratio: ER = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 1)))
    direction[0] = 0  # first element
    
    # 10-period ER
    er_num = np.abs(np.subtract(close, np.roll(close, 10)))
    er_den = np.zeros(n)
    for i in range(n):
        start = max(0, i-9)
        er_den[i] = np.sum(change[start:i+1]) if i >= start else 0
    er = np.divide(er_num, er_den, out=np.zeros_like(er_num), where=er_den!=0)
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly EMA20 for higher timeframe trend
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # CHOP (14-period) on daily data
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            hl = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
            hc = np.abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1])
            lc = np.abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            tr_1d[i] = max(hl, hc, lc)
        atr_1d[i] = np.mean(tr_1d[max(0, i-13):i+1]) if i >= 1 else tr_1d[i]
    
    # CHOP = 100 * log10( sum(ATR14) / (HH-HL) ) / log10(14)
    chop = np.full(len(df_1d), 50.0)  # default to neutral
    for i in range(13, len(df_1d)):
        atr_sum = np.sum(atr_1d[i-13:i+1])
        hh = np.max(df_1d['high'].iloc[i-13:i+1])
        hl = np.min(df_1d['low'].iloc[i-13:i+1])
        if hh > hl:
            chop[i] = 100 * np.log10(atr_sum / (hh - hl)) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike (20-period MA on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), daily EMA (34), weekly EMA (20), CHOP (14), volume MA (20)
    start_idx = max(10, 34, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters: price above/below KAMA + daily EMA34 + weekly EMA20
        uptrend = close[i] > kama[i] and close[i] > ema_34_1d_aligned[i] and close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < kama[i] and close[i] < ema_34_1d_aligned[i] and close[i] < ema_20_1w_aligned[i]
        
        # CHOP regime: > 61.8 = range (mean revert), < 38.2 = trend
        chop_value = chop_aligned[i]
        in_range = chop_value > 61.8
        in_trend = chop_value < 38.2
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # In range: mean reversion at extremes (price vs KAMA)
            # In trend: follow trend with volume confirmation
            if in_range:
                # Mean reversion: long when price < KAMA, short when price > KAMA
                if close[i] < kama[i] * 0.995 and volume_confirm:  # 0.5% below KAMA
                    signals[i] = 0.25
                    position = 1
                elif close[i] > kama[i] * 1.005 and volume_confirm:  # 0.5% above KAMA
                    signals[i] = -0.25
                    position = -1
            elif in_trend:
                # Trend following: follow direction with volume
                if uptrend and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif downtrend and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses above KAMA or trend/chop changes
            if close[i] > kama[i] * 1.002 or not (uptrend if in_trend else True) or (in_range and close[i] > kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below KAMA or trend/chop changes
            if close[i] < kama[i] * 0.998 or not (downtrend if in_trend else True) or (in_range and close[i] < kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals