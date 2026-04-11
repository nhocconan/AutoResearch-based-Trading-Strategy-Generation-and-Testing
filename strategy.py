#!/usr/bin/env python3
# 12h_1d_trix_volume_regime_v1
# Strategy: 12h TRIX(12) signal line + volume confirmation + 1d chop regime filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: TRIX captures momentum, volume confirms strength, and chop filter avoids whipsaws in ranging markets. Works in bull via TRIX>0, in bear via TRIX<0. Low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trix_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Chop regime filter: Chop > 61.8 = range (avoid), Chop < 38.2 = trending (allow)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop[np.isnan(chop) | (atr14 == 0)] = 50
    chop_trending = chop < 38.2  # Trending regime
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # 12h TRIX(12,9) - triple EMA
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(chop_trending_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: TRIX signal + volume + trending regime
        if trix[i] > 0 and vol_confirm[i] and chop_trending_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif trix[i] < 0 and vol_confirm[i] and chop_trending_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite TRIX signal with volume confirmation
        elif position == 1 and (trix[i] < 0) and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix[i] > 0) and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals