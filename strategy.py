#!/usr/bin/env python3
"""
1h_SuperTrend_TrendFollowing_4hFilter
Hypothesis: Supertrend on 1h captures momentum with ATR-based dynamic support/resistance. 
4h EMA50 provides trend filter to avoid counter-trend trades. Volatility filter (ATR ratio) avoids whipsaws in low volatility.
Designed for 1h timeframe with moderate trade frequency (15-30/year) by requiring trend alignment and volatility confirmation.
Works in bull/bear markets by requiring trend alignment and filtering low-volatility periods.
"""

name = "1h_SuperTrend_TrendFollowing_4hFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Supertrend on 1h (ATR=10, multiplier=3.0)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_upper = np.copy(upper_band)
    final_lower = np.copy(lower_band)
    for i in range(1, len(close)):
        if close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = min(upper_band[i], final_upper[i-1])
        if close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = max(lower_band[i], final_lower[i-1])
    
    # Supertrend direction
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    for i in range(1, len(close)):
        if close[i] > final_upper[i-1]:
            direction[i] = 1
        elif close[i] < final_lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
    
    # Volatility filter: ATR ratio (current ATR / 50-period ATR MA) to avoid low volatility
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.divide(atr, atr_ma50, out=np.zeros_like(atr), where=atr_ma50!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(supertrend[i]) or 
            np.isnan(direction[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend using aligned EMA
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volatility filter: require sufficient volatility (ATR ratio > 0.8)
        vol_filter = atr_ratio[i] > 0.8
        
        if position == 0:
            # Long: Supertrend uptrend + 4h uptrend + volatility filter
            if direction[i] == 1 and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Supertrend downtrend + 4h downtrend + volatility filter
            elif direction[i] == -1 and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Supertrend turns down OR 4h trend turns down
            if direction[i] == -1 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Supertrend turns up OR 4h trend turns up
            if direction[i] == 1 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals