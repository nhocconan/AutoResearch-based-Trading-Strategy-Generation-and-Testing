#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(50) trend + volume confirmation
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 1.5x 20-day average
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 1.5x 20-day average
# Exit when price returns to Donchian midpoint or opposite breakout occurs
# Uses weekly trend filter to avoid counter-trend trades in both bull and bear markets
# Target: 30-100 trades over 4 years by requiring confluence of breakout, trend, and volume

name = "1d_donchian_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 1d
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # EMA(50) on 1w
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to initialize
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < Donchian midpoint OR opposite breakout
            if close[i] < donchian_mid[i] or (low[i] < low_20[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > Donchian midpoint OR opposite breakout
            if close[i] > donchian_mid[i] or (high[i] > high_20[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + trend + volume
            # Long: price breaks above upper band AND price > weekly EMA50 AND volume confirmation
            if high[i] > high_20[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND price < weekly EMA50 AND volume confirmation
            elif low[i] < low_20[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
    
    return signals