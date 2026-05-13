#!/usr/bin/env python3
"""
1h_OBV_Trend_Filter
Hypothesis: On-Balance Volume (OBV) confirms trend strength. When OBV makes a new high/low with price confirmation and aligned 4h trend, it signals continuation. 1h timeframe for entry timing, 4h for trend filter. Target: 15-30 trades/year to avoid fee drag. Works in bull/bear via trend filter.
"""

name = "1h_OBV_Trend_Filter"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate OBV
    price_change = np.diff(close, prepend=close[0])
    volume_signed = np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
    obv = np.cumsum(volume_signed)
    
    # 4h trend filter: EMA(50) on close
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip first bar for OBV calculation
        if position == 0:
            # LONG: OBV makes new high, price above 4h EMA50 (uptrend)
            if obv[i] > obv[i-1] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: OBV makes new low, price below 4h EMA50 (downtrend)
            elif obv[i] < obv[i-1] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: OBV makes new low OR price breaks below 4h EMA50
            if obv[i] < obv[i-1] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: OBV makes new high OR price breaks above 4h EMA50
            if obv[i] > obv[i-1] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals