# 12h_KAMA_CCI_Trend_V1
# KAMA (Kaufman Adaptive Moving Average) with CCI (Commodity Channel Index) filter
# Long: Price > KAMA + CCI > 100 (strong uptrend)
# Short: Price < KAMA + CCI < -100 (strong downtrend)
# Exit: Price crosses KAMA in opposite direction
# Uses 12h timeframe with 1d CCI for trend confirmation
# Designed for ~20-40 trades/year to minimize fee drag
# Works in both bull (trend following) and bear (counter-trend reversals via KAMA adaptation)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for CCI calculation (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 12h price (primary timeframe)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            sum_abs_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if sum_abs_change > 0:
                er[i] = price_change / sum_abs_change
            else:
                er[i] = 0
    
    # Smooth ER
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate CCI on 1d data
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_mean = typical_price.rolling(window=20, min_periods=20).mean()
    tp_dev = typical_price.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - tp_mean) / (0.015 * tp_dev)
    cci_values = cci.values
    
    # Align CCI to 12h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after KAMA warmup
        # Get values
        close_val = close[i]
        kama_val = kama[i]
        cci_val = cci_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(kama_val) or 
            np.isnan(cci_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA and CCI > 100 (strong uptrend)
            if close_val > kama_val and cci_val > 100:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA and CCI < -100 (strong downtrend)
            elif close_val < kama_val and cci_val < -100:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below KAMA
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above KAMA
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_KAMA_CCI_Trend_V1
# KAMA adapts to market noise - fast in trends, slow in ranges
# CCI filter ensures we only trade strong trends (>100 or <-100)
# Low frequency: ~20-40 trades/year ideal for 12h timeframe
# Works in bull markets (trend following) and bear markets (KAMA adapts to volatility)
name = "12h_KAMA_CCI_Trend_V1"
timeframe = "12h"
leverage = 1.0