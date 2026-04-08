#!/usr/bin/env python3
"""
6h Linear Regression Channel + 1w Trend Filter + Volume Confirmation
Hypothesis: Price reverting to linear regression trend with volume confirmation captures mean-reversion in trending markets. Uses weekly trend to filter direction and avoid counter-trend trades. Works in both bull/bear by trading mean reversion within the dominant trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_linreg_channel_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def _linreg_channel(arr, window):
    """Linear regression channel: returns upper, lower, midline"""
    n = len(arr)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    midline = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        y = arr[i - window + 1:i + 1]
        x = np.arange(window)
        if np.any(np.isnan(y)):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        y_end = slope * (window - 1) + intercept
        y_start = intercept
        midline[i] = (y_start + y_end) / 2
        dev = np.std(y - (slope * x + intercept))
        upper[i] = midline[i] + dev * 1.5
        lower[i] = midline[i] - dev * 1.5
    return upper, lower, midline

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Linear regression channel (60 periods ~ 15 days on 6h)
    lr_upper, lr_lower, lr_mid = _linreg_channel(close, 60)
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(lr_mid[i]) or 
            np.isnan(lr_upper[i]) or 
            np.isnan(lr_lower[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches midline OR trend turns bearish
            if (close[i] >= lr_mid[i] or 
                close[i] <= ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches midline OR trend turns bullish
            if (close[i] <= lr_mid[i] or 
                close[i] >= ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches lower channel + uptrend + volume spike
            if (close[i] <= lr_lower[i] and
                close[i] >= ema_20_1w_aligned[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches upper channel + downtrend + volume spike
            elif (close[i] >= lr_upper[i] and
                  close[i] <= ema_20_1w_aligned[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals