#!/usr/bin/env python3
"""
4H_KAMA_BREAKOUT_VOLUME_REGIME
Hypothesis: KAMA adapts to market efficiency, reducing lag in trends and noise in ranges.
Long when price breaks above KAMA(10) with volume spike and chop regime (range), 
short when breaks below with volume spike and chop regime.
Exit on opposite KAMA cross. Uses 1d trend filter to avoid counter-trend trades.
Designed to capture breakouts in ranging markets while avoiding whipsaws in strong trends.
Target: 25-40 trades/year with high win rate via volume and regime confirmation.
"""

name = "4H_KAMA_BREAKOUT_VOLUME_REGIME"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - 10 period
    # ER = Efficiency Ratio = |net change| / sum(|abs changes|)
    change = np.abs(close - np.roll(close, 1))
    change[0] = 0
    direction = np.abs(close - np.roll(close, 10))
    # Avoid division by zero
    er = np.where(change.sum() > 0, direction / (np.convolve(change, np.ones(10), 'same') + 1e-10), 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike detection - volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 1.5
    
    # Choppiness Index - chop > 61.8 indicates ranging market
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # True Range average
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop calculation
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    chop[np.isnan(chop)] = 50  # neutral value when undefined
    chop_range = chop > 61.8  # ranging market
    
    # 1d EMA for trend filter - only trade in direction of higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(ema1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price crosses above KAMA with volume spike in ranging market
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and vol_spike[i] and chop_range[i]:
                # Only take long if 1d trend is up or neutral (avoid strong downtrend)
                if close[i] >= ema1d_aligned[i] * 0.98:  # allow small buffer
                    signals[i] = 0.25
                    position = 1
            # SHORT: price crosses below KAMA with volume spike in ranging market
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and vol_spike[i] and chop_range[i]:
                # Only take short if 1d trend is down or neutral (avoid strong uptrend)
                if close[i] <= ema1d_aligned[i] * 1.02:  # allow small buffer
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA or strong trend develops
            if close[i] < kama[i] or close[i] > ema1d_aligned[i] * 1.05:  # exit if strong uptrend develops
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA or strong trend develops
            if close[i] > kama[i] or close[i] < ema1d_aligned[i] * 0.95:  # exit if strong downtrend develops
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals