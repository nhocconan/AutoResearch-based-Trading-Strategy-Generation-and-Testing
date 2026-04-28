#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_1dTrend
Hypothesis: TRIX (12-period) on 12h timeframe captures momentum shifts with reduced lag compared to traditional MACD. Combined with volume spike confirmation and 1d EMA34 trend filter, this strategy aims to capture significant momentum moves while minimizing false signals. The 12h timeframe targets 12-37 trades/year to stay within optimal range, reducing fee drag. TRIX is effective in both bull and bear markets as it identifies accelerating momentum regardless of direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate TRIX on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 - np.roll(ema3, 1)
    trix[0] = 0  # First value has no previous
    trix_12h_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # TRIX signals: positive = bullish momentum, negative = bearish momentum
        trix_bullish = trix_12h_aligned[i] > 0
        trix_bearish = trix_12h_aligned[i] < 0
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions with volume confirmation and trend alignment
        long_entry = trix_bullish and volume_spike[i] and uptrend
        short_entry = trix_bearish and volume_spike[i] and downtrend
        
        # Exit when TRIX changes sign or trend fails
        long_exit = (trix_12h_aligned[i] <= 0) or (not uptrend)
        short_exit = (trix_12h_aligned[i] >= 0) or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_TRIX_VolumeSpike_1dTrend"
timeframe = "12h"
leverage = 1.0