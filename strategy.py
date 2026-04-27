#!/usr/bin/env python3
"""
6h_RelativeStrengthMomentum_1wTrend_1dVol
Hypothesis: Relative strength momentum (RSM) between BTC and ETH on 6h, filtered by 1-week trend and 1-day volume spike, captures leadership shifts in both bull and bear markets. Uses cross-asset momentum as a leading indicator. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week EMA50 for trend filter (bullish if price > EMA50)
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1-day volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Relative Strength Momentum: 6h price change ratio vs BTC (if ETH) or vs ETH (if BTC)
    # We'll compute RSM as the 6-period rate of change (6*6h = 1 day) normalized
    # For simplicity, we use price relative to its 6-period MA as momentum proxy
    close_series = pd.Series(close)
    ma6 = close_series.rolling(window=6, min_periods=6).mean().values
    rsm = (close - ma6) / ma6  # % deviation from 6-period MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA and 6 for rsm
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsm[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: positive RSM (strong momentum) + weekly uptrend + volume spike
            if (rsm[i] > 0.01 and close[i] > ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: negative RSM (weak momentum) + weekly downtrend + volume spike
            elif (rsm[i] < -0.01 and close[i] < ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSM turns negative or trend fails
            if (rsm[i] < 0 or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSM turns positive or trend fails
            if (rsm[i] > 0 or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RelativeStrengthMomentum_1wTrend_1dVol"
timeframe = "6h"
leverage = 1.0