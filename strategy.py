#!/usr/bin/env python3
# 4H_Chaikin_Money_Flow_Touch_1D_Trend
# Hypothesis: Chaikin Money Flow (CMF) at 4h detects accumulation/distribution.
# Long when CMF crosses above 0.25 in 1d uptrend (close > EMA34), short when CMF crosses below -0.25 in 1d downtrend.
# Uses volume-weighted accumulation to identify institutional interest, works in both bull/bear via 1d trend filter.
# Target: 25-40 trades/year with size 0.25 to avoid fee drag.

name = "4H_Chaikin_Money_Flow_Touch_1D_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Chaikin Money Flow (CMF) = (Sum of Money Flow Volume over N) / (Sum of Volume over N)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where(high == low, 0, mfm)  # avoid division by zero
    mfv = mfm * volume
    
    # 20-period CMF
    cmf_num = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    cmf_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(cmf_den != 0, cmf_num / cmf_den, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have CMF data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(cmf[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF crosses above 0.25 + Uptrend (close > EMA34)
            if cmf[i] > 0.25 and cmf[i-1] <= 0.25 and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF crosses below -0.25 + Downtrend (close < EMA34)
            elif cmf[i] < -0.25 and cmf[i-1] >= -0.25 and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CMF falls back below 0 or trend turns down
            if cmf[i] < 0 or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CMF rises back above 0 or trend turns up
            if cmf[i] > 0 or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals