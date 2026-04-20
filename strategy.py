#!/usr/bin/env python3
"""
4h_Volume_Weighted_CCI_Momentum
Hypothesis: Use 4-hour CCI with volume weighting to detect momentum extremes, filtered by daily trend.
Long when volume-weighted CCI crosses above -100 and daily trend is up; short when crosses below +100 and daily trend is down.
Volume weighting reduces false signals in low-liquidity periods. CCI captures cyclical extremes. Daily trend filter ensures alignment with higher timeframe momentum.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: daily filter avoids counter-trend trades, volume-weighted CCI reduces noise.
"""

name = "4h_Volume_Weighted_CCI_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema50_daily[i-1]
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Calculate Volume-Weighted CCI (20-period)
    typical_price = (high + low + close) / 3.0
    vw_tp = typical_price * volume
    
    # Sum of volume-weighted typical price and volume for VWAP
    sum_vw_tp = np.zeros(n)
    sum_volume = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - 19)  # 20-period window
        sum_vw_tp[i] = np.sum(vw_tp[start_idx:i+1])
        sum_volume[i] = np.sum(volume[start_idx:i+1])
    
    vwap = np.divide(sum_vw_tp, sum_volume, out=np.full_like(sum_vw_tp, np.nan), where=sum_volume!=0)
    
    # Mean deviation
    mean_dev = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        if sum_volume[i] > 0:
            vwap_val = vwap[i]
            deviations = np.abs(typical_price[start_idx:i+1] - vwap_val)
            weighted_dev = deviations * volume[start_idx:i+1]
            mean_dev[i] = np.sum(weighted_dev) / sum_volume[i]
        else:
            mean_dev[i] = 0
    
    # CCI calculation
    cci = np.divide((typical_price - vwap), (0.015 * mean_dev), out=np.full_like(typical_price, np.nan), where=(mean_dev!=0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(ema50_daily_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CCI crosses above -100 + daily uptrend
            if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below +100 + daily downtrend
            elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema50_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI crosses below +100 OR daily trend turns down
            if cci[i] < 100 and cci[i-1] >= 100 or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses above -100 OR daily trend turns up
            if cci[i] > -100 and cci[i-1] <= -100 or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals