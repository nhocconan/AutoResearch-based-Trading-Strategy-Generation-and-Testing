#!/usr/bin/env python3
"""
6h_1d_TrendReversal_VolumeSurge_v1
Hypothesis: On 6h timeframe, use 1d trend filter (EMA50) with volume surge and RSI reversal signals.
Long when: price > 1d EMA50 (uptrend), RSI < 30 (oversold), volume > 2x 20-period average.
Short when: price < 1d EMA50 (downtrend), RSI > 70 (overbought), volume > 2x 20-period average.
This captures mean reversion within the trend, filtering counter-trend noise.
Volume surge confirms institutional participation in the reversal.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drift.
Works in bull via pullback longs in uptrend, in bear via bounce shorts in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_TrendReversal_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral if not enough data
    
    # Volume surge: current volume > 2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend and reversal conditions
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        volume_surge = vol_ratio[i] > 2.0
        
        # Entry conditions
        long_entry = uptrend and oversold and volume_surge
        short_entry = downtrend and overbought and volume_surge
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        long_exit = rsi[i] > 40
        short_exit = rsi[i] < 60
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals