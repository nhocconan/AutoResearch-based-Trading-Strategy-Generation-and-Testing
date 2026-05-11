#!/usr/bin/env python3
"""
4H_LinearRegression_Trend_1dATR_Support
Hypothesis: Linear regression slope on 4h close defines trend, 1d ATR-based support/resistance provides dynamic entry/exit zones.
Long when slope > 0 and price bounces above support; short when slope < 0 and price rejects below resistance.
Uses 4h timeframe to balance trade frequency and reduce fee drag. Designed to work in both bull and bear trends.
"""

name = "4H_LinearRegression_Trend_1dATR_Support"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for ATR-based support/resistance
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic support/resistance: close ± 1.5 * ATR
    support = close_1d - 1.5 * atr14
    resistance = close_1d + 1.5 * atr14
    
    # Align to 4h timeframe
    support_aligned = align_htf_to_ltf(prices, df_1d, support)
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance)
    
    # 4h linear regression slope (20-period)
    def linreg_slope(arr, window):
        slopes = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes[i] = slope
        return slopes
    
    lr_slope = linreg_slope(close, 20)
    
    # Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(lr_slope[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(resistance_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        trend_up = lr_slope[i] > 0
        trend_down = lr_slope[i] < 0
        bounce_support = close[i] > support_aligned[i] and close[i-1] <= support_aligned[i-1]
        reject_resistance = close[i] < resistance_aligned[i] and close[i-1] >= resistance_aligned[i-1]
        
        if position == 0:
            # Long: Uptrend + bounce off support + volume spike
            if trend_up and bounce_support and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Downtrend + rejection at resistance + volume spike
            elif trend_down and reject_resistance and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: trend reversal OR break below support
                if not trend_up or close[i] < support_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: trend reversal OR break above resistance
                if not trend_down or close[i] > resistance_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals