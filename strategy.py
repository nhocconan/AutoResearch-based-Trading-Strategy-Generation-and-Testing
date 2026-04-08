#!/usr/bin/env python3
"""
1d_1w_bollinger_squeeze_breakout
Hypothesis: Use 1d Bollinger Band squeeze for volatility compression, breakout with 1w trend filter.
Long when price breaks above upper BB during 1w uptrend, short when breaks below lower BB during 1w downtrend.
Volatility squeeze reduces false breakouts. Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend).
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_bollinger_squeeze_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(21) for trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std_dev = close_series.rolling(window=bb_window, min_periods=bb_window).std().values
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # Bollinger Band width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < bb_width_ma * 0.8  # Bollinger Band squeeze
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(squeeze_condition[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below middle band or trend changes to downtrend
            if close[i] < bb_middle[i] or ema_1w_aligned[i] < ema_1w_aligned[max(0, i-5)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band or trend changes to uptrend
            if close[i] > bb_middle[i] or ema_1w_aligned[i] > ema_1w_aligned[max(0, i-5)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above upper BB during squeeze and 1w uptrend
            if (close[i] > bb_upper[i] and 
                squeeze_condition[i] and 
                ema_1w_aligned[i] > ema_1w_aligned[max(0, i-5)]):  # Uptrend confirmation
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower BB during squeeze and 1w downtrend
            elif (close[i] < bb_lower[i] and 
                  squeeze_condition[i] and 
                  ema_1w_aligned[i] < ema_1w_aligned[max(0, i-5)]):  # Downtrend confirmation
                position = -1
                signals[i] = -0.25
    
    return signals