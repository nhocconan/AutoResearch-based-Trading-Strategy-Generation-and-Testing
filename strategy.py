#!/usr/bin/env python3
"""
12h_1dDonchian20_Breakout_1dTrend_Volume
Hypothesis: Breakout above/below daily Donchian(20) on 12h timeframe with daily EMA50 trend filter and volume confirmation.
Daily Donchian provides clear support/resistance levels. Breakouts in direction of daily trend with volume
should capture strong momentum moves. Works in bull/bear markets by aligning with daily trend direction.
"""

name = "12h_1dDonchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Donchian Channel (20) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high and low for Donchian
    dh_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    dh_20_12h = align_htf_to_ltf(prices, df_1d, dh_20)
    dl_20_12h = align_htf_to_ltf(prices, df_1d, dl_20)
    
    # === Daily Trend Filter (EMA50) ===
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Filter (1.8x 20-period EMA on 12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian calculation)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(dh_20_12h[i]) or np.isnan(dl_20_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above daily Donchian high with uptrend and volume
            if (close[i] > dh_20_12h[i] and 
                close[i] > ema50_12h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below daily Donchian low with downtrend and volume
            elif (close[i] < dl_20_12h[i] and 
                  close[i] < ema50_12h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below daily Donchian low (mean reversion)
            if close[i] < dl_20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above daily Donchian high (mean reversion)
            if close[i] > dh_20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals