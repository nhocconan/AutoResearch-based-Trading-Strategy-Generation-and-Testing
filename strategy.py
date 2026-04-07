#!/usr/bin/env python3
"""
4h_cci_trend_reversal_v1
Hypothesis: On 4h timeframe, enter long when CCI crosses above -100 (bullish momentum) with above-average volume and price above 50-period EMA, enter short when CCI crosses below +100 (bearish momentum) with above-average volume and price below 50-period EMA. Exit when CCI crosses zero (momentum exhaustion). Uses 1d CCI trend filter to avoid counter-trend trades. Designed for 20-50 trades/year to minimize fee drag while capturing momentum reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_trend_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h CCI (20-period)
    if len(close) < 20:
        return np.zeros(n)
    
    # Typical Price
    tp = (high + low + close) / 3.0
    
    # Moving Average of TP
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    
    # Mean Deviation
    md = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # CCI
    cci = (tp - ma_tp) / (0.015 * md)
    
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d CCI for trend filter (avoid counter-trend trades)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Typical Price
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # 1d MA of TP
    ma_tp_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d Mean Deviation
    md_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # 1d CCI
    cci_1d = (tp_1d - ma_tp_1d) / (0.015 * md_1d)
    
    # Align indicators to 4h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(cci_1d_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero (momentum exhaustion)
            if cci[i] < 0 and cci[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero (momentum exhaustion)
            if cci[i] > 0 and cci[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: CCI crosses above -100 with price above EMA50 and 1d CCI bullish
                if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema_50[i] and cci_1d_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short: CCI crosses below +100 with price below EMA50 and 1d CCI bearish
                elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema_50[i] and cci_1d_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals