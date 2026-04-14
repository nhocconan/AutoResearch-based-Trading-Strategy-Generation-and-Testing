#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour RSI(14) mean reversion with 1-day Bollinger Bands filter and volume confirmation.
# In bear markets (2025+), RSI extremes often precede mean-reversion bounces rather than continuation.
# The 1-day Bollinger Bands (20,2) define the medium-term range; trades fade extremes toward the mean.
# Volume > 1.3x average confirms participation at turning points.
# Exit when RSI returns to neutral (40-60) or price touches the opposite Bollinger Band.
# This strategy targets 20-30 trades per year per symbol (80-120 total over 4 years), well within limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Bollinger Bands filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Bollinger Bands (20,2)
    bb_len = 20
    bb_std = 2
    if len(df_1d) < bb_len:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    basis = pd.Series(close_1d).rolling(window=bb_len, min_periods=bb_len).mean().values
    dev = bb_std * pd.Series(close_1d).rolling(window=bb_len, min_periods=bb_len).std().values
    upper_bb = basis + dev
    lower_bb = basis - dev
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # RSI(14) on 4h
    rsi_len = 14
    if n < rsi_len + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(rsi_len + 1, bb_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: RSI oversold (<30) + price near lower BB + volume
            if (rsi[i] < 30 and 
                close[i] <= lower_bb_aligned[i] * 1.02 and  # within 2% of lower BB
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: RSI overbought (>70) + price near upper BB + volume
            elif (rsi[i] > 70 and 
                  close[i] >= upper_bb_aligned[i] * 0.98 and  # within 2% of upper BB
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or price touches upper BB
            if (rsi[i] >= 40 and rsi[i] <= 60) or close[i] >= upper_bb_aligned[i] * 0.98:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or price touches lower BB
            if (rsi[i] >= 40 and rsi[i] <= 60) or close[i] <= lower_bb_aligned[i] * 1.02:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI_BB_Volume_MeanRev_v1"
timeframe = "4h"
leverage = 1.0