#!/usr/bin/env python3
# 4h_VWAP_MeanReversion_With_1dTrend_Filter
# Hypothesis: Price reverts to VWAP in ranging markets but trends with VWAP in trending markets.
# We use 1d EMA50 as trend filter to avoid counter-trend trades. In uptrend, buy when price
# crosses below VWAP; in downtrend, sell when price crosses above VWAP. VWAP calculated
# using typical price and volume. This strategy aims to capture mean reversion in ranges
# and pullbacks in trends, with low trade frequency to minimize fee drag.

name = "4h_VWAP_MeanReversion_With_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50) and enough data for cumulative VWAP (use 20 bars min)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if trend filter is NaN
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: uptrend + price crosses below VWAP (mean reversion long)
            if uptrend and close[i] < vwap[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price crosses above VWAP (mean reversion short)
            elif downtrend and close[i] > vwap[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price crosses back above VWAP
            if not uptrend or close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price crosses back below VWAP
            if not downtrend or close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals