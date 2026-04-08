# 6h_1d_cci_trend_reversal_v1
# Hypothesis: 6-hour CCI mean reversion with 1-day trend filter.
# In ranging markets, price reverts from CCI extremes (>100 or <-100).
# In trending markets (price > 1d EMA50 for long, < 1d EMA50 for short),
# we allow CCI to pull back to zero for trend continuation.
# Uses 1d EMA50 for trend filter and 6h CCI for entry/exit.
# Target: 15-30 trades/year to minimize fee drag while capturing reversals and pullbacks.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_trend_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h CCI(20)
    typical_price = (high + low + close) / 3.0
    sma_tp = np.zeros(n)
    mad = np.zeros(n)
    
    for i in range(20, n):
        sma_tp[i] = np.mean(typical_price[i-19:i+1])
        mad[i] = np.mean(np.abs(typical_price[i-19:i+1] - sma_tp[i]))
    
    cci = np.zeros(n)
    for i in range(20, n):
        if mad[i] != 0:
            cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mad[i])
        else:
            cci[i] = 0.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.zeros(len(close_1d))
    ema_1d_50[:] = np.nan
    ema_1d_50[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema_1d_50[i] = close_1d[i] * 0.0377 + ema_1d_50[i-1] * 0.9623
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        cci_val = cci[i]
        ema_1d = ema_1d_50_aligned[i]
        price = close[i]
        
        if np.isnan(ema_1d) or np.isnan(cci_val):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: CCI crosses below zero (end of pullback/reversion)
            if cci_val < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: CCI crosses above zero (end of pullback/reversion)
            if cci_val > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            # Long: CCI < -100 (oversold) AND price > 1d EMA50 (uptrend)
            # Short: CCI > 100 (overbought) AND price < 1d EMA50 (downtrend)
            if cci_val < -100 and price > ema_1d:
                position = 1
                signals[i] = 0.25
            elif cci_val > 100 and price < ema_1d:
                position = -1
                signals[i] = -0.25
    
    return signals