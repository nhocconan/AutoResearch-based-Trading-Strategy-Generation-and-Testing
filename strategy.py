#!/usr/bin/env python3
# 1h_4h1d_Momentum_Reversal
# Hypothesis: In both bull and bear markets, price often reverses from intraday extremes.
# Uses 1d RSI(14) for medium-term trend bias (RSI>50 = long bias, RSI<50 = short bias).
# Enters on 1h mean reversion when price touches Bollinger Bands (20,2) in direction of 1d trend.
# Filters with volume spike (1.5x 24-period MA) to avoid false signals in low volume.
# Targets 15-30 trades/year by requiring 1d trend alignment + BB touch + volume confirmation.

name = "1h_4h1d_Momentum_Reversal"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend bias (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d RSI(14) for trend bias
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # Not enough data for first 13 periods
    
    # Align 1d RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Bollinger Bands (20,2) on 1h
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    bb_upper = bb_upper.values
    bb_lower = bb_lower.values
    
    # Volume average (24-period for 1h = 24 hours)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for BB (20) + vol MA (24) + RSI alignment
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend bias: RSI > 50 = long bias, RSI < 50 = short bias
        long_bias = rsi_1d_aligned[i] > 50
        short_bias = rsi_1d_aligned[i] < 50
        
        # Volume confirmation (1.5x average for significance)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price touches lower BB in 1d uptrend bias with volume spike
            if low[i] <= bb_lower[i] and long_bias and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: Price touches upper BB in 1d downtrend bias with volume spike
            elif high[i] >= bb_upper[i] and short_bias and volume_surge:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Long exit: price returns to middle BB or 1d bias flips
                if close[i] >= bb_mid.iloc[i] or not long_bias:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: price returns to middle BB or 1d bias flips
                if close[i] <= bb_mid.iloc[i] or not short_bias:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals