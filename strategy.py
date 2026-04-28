#!/usr/bin/env python3
"""
6h_RSI_4H_Slope_PriceAction
Hypothesis: On 6h timeframe, use 4-hour RSI slope to detect momentum exhaustion in strong trends, combined with price action rejection at Bollinger Bands on 6h. This captures mean reversion in overextended moves while avoiding chop. Works in both bull (fade rallies) and bear (fade crashes) by fading momentum extremes. Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for RSI slope calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate RSI(14) on 4h close
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Calculate RSI slope (3-period linear regression slope)
    def rolling_slope(arr, window):
        slopes = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window-1, len(arr)):
            y = arr[i-window+1:i+1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                continue
            # Use only valid points
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            slopes[i] = slope
        return slopes
    
    rsi_slope = rolling_slope(rsi_4h, 3)
    rsi_slope_aligned = align_htf_to_ltf(prices, df_4h, rsi_slope)
    
    # Bollinger Bands on 6h (20, 2)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2 * std_20)
    lower_bb = ma_20 - (2 * std_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_slope_aligned[i]) or 
            np.isnan(ma_20[i]) or
            np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        # RSI slope conditions: negative slope = bearish momentum, positive = bullish
        rsi_slope_neg = rsi_slope_aligned[i] < -0.15  # weakening momentum
        rsi_slope_pos = rsi_slope_aligned[i] > 0.15   # strengthening momentum
        
        # Price action rejection at Bollinger Bands
        near_upper = close[i] > upper_bb[i] * 0.998  # within 0.2% of upper band
        near_lower = close[i] < lower_bb[i] * 1.002  # within 0.2% of lower band
        
        # Entry logic: fade momentum extremes at BB
        long_entry = rsi_slope_neg and near_lower  # weakening bearish momentum at lower BB
        short_entry = rsi_slope_pos and near_upper  # weakening bullish momentum at upper BB
        
        # Exit logic: momentum resets or price moves to middle
        long_exit = rsi_slope_pos or close[i] > ma_20[i]
        short_exit = rsi_slope_neg or close[i] < ma_20[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_RSI_4H_Slope_PriceAction"
timeframe = "6h"
leverage = 1.0