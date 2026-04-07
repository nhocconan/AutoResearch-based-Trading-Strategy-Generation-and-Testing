#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h KAMA Trend + 1w Momentum + Volume Filter
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# Combined with 1w momentum (rate of change) to ensure we only trade in strong trends,
# and volume confirmation to avoid low-liquidity breakouts. Works in bull via upward KAMA
# slope + positive momentum, in bear via downward KAMA slope + negative momentum.
# Volume filter ensures institutional participation. Target: 12-37 trades/year.

name = "6h_kama_trend_1w_momentum_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA components (ER = Efficiency Ratio, SC = Smoothing Constant)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smooth ER over 10 periods
    er_smoothed = pd.Series(er).ewm(span=10, adjust=False).mean().values
    
    # Smoothing constants
    sc = (er_smoothed * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1w momentum: ROC(12) - 3-period average for smoothing
    roc_12 = np.zeros_like(close_1w)
    for i in range(12, len(close_1w)):
        roc_12[i] = (close_1w[i] - close_1w[i-12]) / close_1w[i-12] * 100
    roc_smoothed = pd.Series(roc_12).rolling(window=3, min_periods=1).mean().values
    roc_1w_aligned = align_htf_to_ltf(prices, df_1w, roc_smoothed)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(roc_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_condition = vol_ok[i]
        
        if position == 1:  # Long position
            # Exit: KAMA turns downward or momentum turns negative
            if close[i] < kama[i] or roc_1w_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: KAMA turns upward or momentum turns positive
            if close[i] > kama[i] or roc_1w_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_condition:
                # Bullish: price above KAMA + positive momentum
                if close[i] > kama[i] and roc_1w_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Bearish: price below KAMA + negative momentum
                elif close[i] < kama[i] and roc_1w_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals