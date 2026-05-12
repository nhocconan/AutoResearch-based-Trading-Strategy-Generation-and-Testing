# 6h_ChandelierTrend_Retracement
# Trend-following with dynamic stop: Chandelier Exit for exit, price retracement to EMA for entry.
# Works in bull by catching breakouts, in bear by shorting retracements in downtrends.
# Uses 6h EMA21 trend filter and Chandelier(22,3) for exits. Low frequency (<30/year).
name = "6h_ChandelierTrend_Retracement"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA21 trend filter
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Chandelier Exit (22,3): uses ATR(22)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr22 = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    highest_high_22 = pd.Series(high).rolling(window=22, min_periods=22).max().values
    lowest_low_22 = pd.Series(low).rolling(window=22, min_periods=22).min().values
    long_stop = highest_high_22 - 3 * atr22
    short_stop = lowest_low_22 + 3 * atr22
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(ema21[i]) or np.isnan(long_stop[i]) or np.isnan(short_stop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > EMA21 (uptrend) and retracing to EMA21 from above
            if close[i] > ema21[i] and close[i] <= ema21[i] * 1.005:  # within 0.5% above EMA
                signals[i] = 0.25
                position = 1
            # SHORT: price < EMA21 (downtrend) and retracing to EMA21 from below
            elif close[i] < ema21[i] and close[i] >= ema21[i] * 0.995:  # within 0.5% below EMA
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price hits Chandelier long stop
            if close[i] <= long_stop[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price hits Chandelier short stop
            if close[i] >= short_stop[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals