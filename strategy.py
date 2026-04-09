#!/usr/bin/env python3
# 6h_cci_trend_follow_v1
# Hypothesis: 6h strategy using CCI(20) for trend strength and reversal signals, with 1d EMA(50) as HTF trend filter.
# Long when CCI crosses above -100 from below AND price > 1d EMA50 (uptrend regime).
# Short when CCI crosses below +100 from above AND price < 1d EMA50 (downtrend regime).
# Exit when CCI crosses back through zero in opposite direction.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-25 trades/year (50-100 total over 4 years) on BTC/ETH/SOL.
# Works in bull markets via trend continuation and bear markets via mean reversion at extremes.
# CCI captures overbought/oversold conditions while respecting the primary trend from 1d EMA.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # CCI calculation: Typical Price = (H+L+C)/3
    typical_price = (high + low + close) / 3.0
    tp_s = pd.Series(typical_price)
    sma_tp = tp_s.rolling(window=20, min_periods=20).mean()
    mad = tp_s.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_s - sma_tp) / (0.015 * mad)
    cci = cci.values
    
    # Load 1d EMA50 as HTF trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(cci[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero
            if cci[i] < 0 and cci[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero
            if cci[i] > 0 and cci[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for CCI crosses at ±100 with HTF trend alignment
            bullish_setup = (cci[i] > -100 and cci[i-1] <= -100) and (close[i] > ema_50_1d_aligned[i])
            bearish_setup = (cci[i] < 100 and cci[i-1] >= 100) and (close[i] < ema_50_1d_aligned[i])
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals