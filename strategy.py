# 4h_RVIX_Trend_Filter
# Hypothesis: Use a relative volatility index (RVIX) to identify high-probability mean-reversion zones filtered by 1-day trend. RVIX > 80 indicates extreme volatility contraction (mean reversion setup), RVIX < 20 indicates expansion (trend continuation). Only trade in direction of 1-day EMA50 to avoid counter-trend moves. Designed for 20-40 trades/year with clear entry/exit rules to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === True Range and ATR(14) on 4h ===
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === RVIX: (Current TR / ATR(14)) * 100 ===
    # Measures volatility relative to recent average
    rvix = (tr / (atr + 1e-10)) * 100
    
    # === 1-day EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers ATR and 1d EMA
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rvix[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: price above/below 1-day EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: RVIX > 80 (volatility contraction) + uptrend
            if rvix[i] > 80 and uptrend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RVIX > 80 (volatility contraction) + downtrend
            elif rvix[i] > 80 and downtrend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: mean reversion complete when RVIX returns to normal (< 50)
        elif position == 1:
            if rvix[i] < 50:  # volatility normalized, exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if rvix[i] < 50:  # volatility normalized, exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RVIX_Trend_Filter"
timeframe = "4h"
leverage = 1.0