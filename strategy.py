#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-week RSI mean reversion and 1-day volume confirmation.
# In overbought/oversold conditions (weekly RSI >70 or <30), price tends to revert to the mean.
# Uses daily volume spike (volume > 1.5x 20-day average) to confirm the reversal.
# Enters long when weekly RSI <30 and daily volume spike, short when weekly RSI >70 and daily volume spike.
# Exits when weekly RSI returns to neutral range (40-60).
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WeeklyRSI_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate daily volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume']
    vol_ma_20 = volume_1d.rolling(window=20, min_periods=20).mean()
    vol_ma_20_values = vol_ma_20.values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_values)
    
    # Volume spike: current daily volume > 1.5x 20-day average
    volume_spike = volume > 1.5 * vol_ma_20_aligned
    
    # RSI conditions
    rsi_overbought = rsi_1w_aligned > 70
    rsi_oversold = rsi_1w_aligned < 30
    rsi_neutral = (rsi_1w_aligned >= 40) & (rsi_1w_aligned <= 60)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold RSI + volume spike
            if rsi_oversold[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought RSI + volume spike
            elif rsi_overbought[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral range
            if rsi_neutral[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral range
            if rsi_neutral[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals