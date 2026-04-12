#!/usr/bin/env python3
"""
6h_1w_1d_WilliamsVixFix_v1
Hypothesis: On 6h timeframe, use Williams Vix Fix (WVF) from weekly data to detect volatility extremes.
WVF > 0.8 signals high fear (potential bottom) with 1d uptrend filter for longs.
WVF < 0.2 signals low fear (potential top) with 1d downtrend filter for shorts.
Exit when WVF returns to neutral range (0.4-0.6). Designed for low trade frequency by requiring
extreme volatility readings and trend alignment. Works in bull/bear via volatility regime detection.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_WilliamsVixFix_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY WILLIAMS VIX FIX ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Vix Fix: measures volatility as percentage off weekly high
    # Formula: ((Highest Close in period - Low) / Highest Close in period) * 100
    # We use 22-week lookback for highest close (approx 6 months)
    highest_close = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        start_idx = max(0, i - 21)  # 22 periods including current
        highest_close[i] = np.max(close_1w[start_idx:i+1])
    
    # Avoid division by zero
    wvf = np.where(highest_close > 0, ((highest_close - low_1w) / highest_close) * 100, 0)
    # Normalize to 0-1 range (typical WVF ranges 0-100, we scale to 0-1)
    wvf = np.clip(wvf / 100, 0, 1)
    
    # === DAILY EMA(20) FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 20:
        ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_20_1d = np.full_like(close_1d, np.nan)
    
    # Align data to 6h timeframe
    wvf_aligned = align_htf_to_ltf(prices, df_1w, wvf)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(wvf_aligned[i]) or np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        wvf_val = wvf_aligned[i]
        ema_val = ema_20_1d_aligned[i]
        
        # Entry conditions
        # Long: Extreme fear (WVF > 0.8) with price above daily EMA (uptrend)
        long_entry = (wvf_val > 0.8) and (close[i] > ema_val)
        # Short: Extreme complacency (WVF < 0.2) with price below daily EMA (downtrend)
        short_entry = (wvf_val < 0.2) and (close[i] < ema_val)
        
        # Exit conditions: Return to neutral volatility range
        exit_long = wvf_val < 0.6  # Fear subsided
        exit_short = wvf_val > 0.4  # Complacency ended
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals