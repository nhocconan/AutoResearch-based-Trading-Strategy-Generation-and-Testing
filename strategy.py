#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# Long when price breaks above 20-period 6h high AND 12h close > 12h EMA50 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below 20-period 6h low AND 12h close < 12h EMA50 (downtrend) AND volume > 1.5x 20 EMA
# Uses 6h for structure and breakout detection, 12h for trend direction to avoid counter-trend trades.
# Discrete sizing (0.25) to balance return and fee drag. Target: 12-37 trades/year on 6h.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Volume confirmation reduces false breakouts. Trend filter improves win rate in choppy markets.

name = "6h_Donchian20_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = max(high_6h over last 20 periods)
    # Lower band = min(low_6h over last 20 periods)
    high_roll_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align 6h Donchian levels to 6h timeframe (already aligned, but using for consistency)
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, high_roll_20)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, low_roll_20)
    
    # Get 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 6h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_6h_aligned[i]) or np.isnan(lower_6h_aligned[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND 12h uptrend AND volume spike
            if (close[i] > upper_6h_aligned[i] and 
                uptrend_12h_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND 12h downtrend AND volume spike
            elif (close[i] < lower_6h_aligned[i] and 
                  downtrend_12h_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band OR 12h trend changes to downtrend
            if (close[i] < lower_6h_aligned[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band OR 12h trend changes to uptrend
            if (close[i] > upper_6h_aligned[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals