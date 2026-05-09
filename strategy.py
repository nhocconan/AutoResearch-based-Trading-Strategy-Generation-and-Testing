#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-week RSI mean reversion and 1-day volume confirmation.
# In overbought/oversold conditions (weekly RSI >70 or <30), price tends to mean-revert.
# Enters short when weekly RSI >70 and 1-day volume > 1.5x 20-period average.
# Enters long when weekly RSI <30 and 1-day volume > 1.5x 20-period average.
# Exits when weekly RSI returns to neutral range (40-60).
# Uses tight entry conditions to limit trades to 50-150 over 4 years.

name = "12h_WeeklyRSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate 1-day volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume']
    vol_ma_20 = volume_1d.rolling(window=20, min_periods=20).mean()
    vol_ma_20_values = vol_ma_20.values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_values)
    
    # Volume confirmation: current 1-day volume > 1.5x 20-period average
    volume_ratio = volume_1d / vol_ma_20
    volume_ratio_values = volume_ratio.values
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or
            np.isnan(volume_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter short: overbought (RSI > 70) + high volume
            if rsi_1w_aligned[i] > 70 and volume_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
            # Enter long: oversold (RSI < 30) + high volume
            elif rsi_1w_aligned[i] < 30 and volume_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: RSI returns to neutral range (40-60)
            if 40 <= rsi_1w_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral range (40-60)
            if 40 <= rsi_1w_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals