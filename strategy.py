#!/usr/bin/env python3
"""
4h_cci_trend_following_v1
Hypothesis: On 4-hour timeframe, use Commodity Channel Index (CCI) with period 20 to identify overbought/oversold conditions, filtered by daily trend direction. Enter long when CCI crosses below -100 (oversold) in a daily uptrend, and short when CCI crosses above +100 (overbought) in a daily downtrend. Exit when CCI returns to zero line. This captures mean-reversion within the trend, reducing whipsaws. Target: 50-150 trades over 4 years (12-38/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_trend_following_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # CCI parameters
    cci_period = 20
    cci_constant = 0.015
    
    # Calculate Typical Price
    tp = (high + low + close) / 3
    
    # Calculate SMA of TP
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=cci_period, min_periods=cci_period).mean().values
    
    # Calculate Mean Deviation
    md = tp_series.rolling(window=cci_period, min_periods=cci_period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # Calculate CCI
    cci = (tp - sma_tp) / (cci_constant * md)
    # Handle division by zero or near-zero MD
    cci = np.where(md == 0, 0, cci)
    
    # Load daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < cci_period:
        return np.zeros(n)
    
    # Calculate daily EMA for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(cci_period, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend filter
        is_uptrend = daily_uptrend_aligned[i] == 1
        
        if position == 1:  # Long position
            # Exit: CCI crosses above zero (mean reversion complete)
            if cci[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below zero (mean reversion complete)
            if cci[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter if daily trend aligns
            if is_uptrend:
                # Long entry: CCI crosses below -100 (oversold in uptrend)
                if cci[i] < -100 and cci[i-1] >= -100:
                    position = 1
                    signals[i] = 0.25
            else:
                # Short entry: CCI crosses above +100 (overbought in downtrend)
                if cci[i] > 100 and cci[i-1] <= 100:
                    position = -1
                    signals[i] = -0.25
    
    return signals