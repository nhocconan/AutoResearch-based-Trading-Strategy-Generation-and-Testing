#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d KAMA adaptive trend filter with Bollinger Band mean reversion entries
# Long when price crosses below BB lower band AND 1d KAMA trend is up (adaptive smoothing confirms uptrend)
# Short when price crosses above BB upper band AND 1d KAMA trend is down
# Exit when price crosses BB midpoint (20-period SMA)
# Uses discrete sizing 0.25 to minimize fee churn while maintaining adequate position sizing
# KAMA adapts to market noise - fast in trends, slow in ranging markets - ideal for BTC/ETH
# Bollinger Bands provide dynamic support/resistance that volatility-adjusts
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_1dKAMA_BB_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for KAMA
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    # ER = |Change| / Sum(|Changes|) over period
    # Smooth = ER * (fastest - slowest) + slowest
    # Alpha = Smooth^2
    # KAMA = prev_KAMA + Alpha * (price - prev_KAMA)
    er_period = 10
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = pd.Series(change).rolling(window=er_period, min_periods=er_period).sum().values
    price_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.where(volatility > 0, price_change / volatility, 0)
    smooth = er * (fast_sc - slow_sc) + slow_sc
    alpha = smooth ** 2
    
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + alpha[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 6h timeframe (wait for completed 1d bar)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate Bollinger Bands on 6h (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + (bb_std * std_20)
    bb_lower = sma_20 - (bb_std * std_20)
    bb_mid = sma_20  # 20-period SMA as exit level
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses below BB lower band AND 1d KAMA trend is up
            if (close[i] < bb_lower[i] and close[i-1] >= bb_lower[i-1] and 
                kama_aligned[i] > kama_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses above BB upper band AND 1d KAMA trend is down
            elif (close[i] > bb_upper[i] and close[i-1] <= bb_upper[i-1] and 
                  kama_aligned[i] < kama_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above BB midpoint (20-period SMA)
            if close[i] > bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below BB midpoint (20-period SMA)
            if close[i] < bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals