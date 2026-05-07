#!/usr/bin/env python3
"""
6h_RSI_40_60_With_Volume_Spike_and_1dTrend
Hypothesis: RSI between 40-60 indicates low momentum/choppy market, allowing mean reversion.
Enter long when RSI < 40 with volume spike in uptrend (price > 1d EMA50).
Enter short when RSI > 60 with volume spike in downtrend (price < 1d EMA50).
Volume spike = 2x average volume. Uses 1d EMA50 for trend filter.
Designed for 6h timeframe with low trade frequency (~15-30/year) to avoid fee drag.
Works in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI_40_60_With_Volume_Spike_and_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need sufficient warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume spike: at least 2x average volume
        volume_spike = vol_ratio[i] >= 2.0
        
        if position == 0:
            # Long: RSI < 40 (oversold) in uptrend with volume spike
            long_entry = (rsi[i] < 40) and uptrend and volume_spike
            # Short: RSI > 60 (overbought) in downtrend with volume spike
            short_entry = (rsi[i] > 60) and downtrend and volume_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 60 (overbought) or trend changes to downtrend
            if (rsi[i] > 60) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 40 (oversold) or trend changes to uptrend
            if (rsi[i] < 40) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals