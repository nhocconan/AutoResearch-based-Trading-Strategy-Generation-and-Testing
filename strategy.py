# SPDX-FileCopyrightText: 2025 Alpaca Trading Systems
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""
Strategy: 6h_1d_CCI_Trend_Filter
Hypothesis: Use CCI(20) on daily timeframe to identify overbought/oversold conditions,
combined with 60-period EMA trend filter on 6h chart. In ranging markets (CCI between -100 and +100),
we fade extremes; in trending markets (CCI outside range), we follow the trend.
This adapts to both bull and bear regimes by using CCI as a regime filter.
Target: 50-150 trades over 4 years (12-37/year) with disciplined entries.
"""

name = "6h_1d_CCI_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily timeframe
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp = typical_price.rolling(window=20, min_periods=20).mean()
    mad = typical_price.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - sma_tp) / (0.015 * mad)
    cci_values = cci.values
    
    # Daily trend filter: EMA60 > EMA120 for uptrend
    ema60_1d = pd.Series(df_1d['close']).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema120_1d = pd.Series(df_1d['close']).ewm(span=120, adjust=False, min_periods=120).mean().values
    trend_up_1d = ema60_1d > ema120_1d
    trend_down_1d = ema60_1d < ema120_1d
    
    # Align daily indicators to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_values)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # 6h EMA20 for entry timing
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 60)  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(cci_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i]) or
            np.isnan(ema20_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: CCI > -100 (not deeply oversold) + price above EMA20 + daily uptrend
            if (cci_aligned[i] > -100 and 
                close[i] > ema20_6h[i] and 
                trend_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: CCI < 100 (not deeply overbought) + price below EMA20 + daily downtrend
            elif (cci_aligned[i] < 100 and 
                  close[i] < ema20_6h[i] and 
                  trend_down_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI drops below -100 (deep oversold) or trend breaks
            if (cci_aligned[i] < -100 or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI rises above 100 (deep overbought) or trend breaks
            if (cci_aligned[i] > 100 or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals