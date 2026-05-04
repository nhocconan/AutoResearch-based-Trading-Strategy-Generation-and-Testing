#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above upper BB AND 1d close > 1d EMA50 (uptrend) AND volume > 2.0x 20 EMA
# Short when price breaks below lower BB AND 1d close < 1d EMA50 (downtrend) AND volume > 2.0x 20 EMA
# Exit when price reverts to middle BB or trend changes
# Uses Bollinger Bands (20,2.0) for volatility-based breakouts that work in both bull and bear markets
# Volume spike filter reduces false breakouts. Target: 15-35 trades/year.
# Discrete sizing (0.25) to minimize fee churn while maintaining profitability.

name = "12h_BB20_2_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20,2.0) for volatility context
    close_1d = df_1d['close'].values
    bb_ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma_20 + (bb_std_20 * 2.0)
    bb_lower = bb_ma_20 - (bb_std_20 * 2.0)
    
    # Align 1d Bollinger Bands to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_ma_20)
    
    # Get 1d data for EMA50 trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB AND 1d uptrend AND volume spike
            if (close[i] > bb_upper_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB AND 1d downtrend AND volume spike
            elif (close[i] < bb_lower_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to middle BB OR 1d trend changes to downtrend
            if (close[i] < bb_middle_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to middle BB OR 1d trend changes to uptrend
            if (close[i] > bb_middle_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals