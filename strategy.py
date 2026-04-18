#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_DailyEMA_WeeklyTrend_RatioFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate EMA50 on daily close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align to 6h with proper delay (wait for daily close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly EMA20 for trend
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Align weekly to 6h with proper delay
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Price ratio filter: current price vs weekly EMA (normalized distance)
    # Avoid division by zero
    ratio = np.where(ema20_1w_aligned != 0, close / ema20_1w_aligned, 1.0)
    # Smooth the ratio to reduce noise
    ratio_smooth = pd.Series(ratio).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or
            np.isnan(ratio_smooth[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50_val = ema50_1d_aligned[i]
        ema20w_val = ema20_1w_aligned[i]
        ratio_val = ratio_smooth[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long conditions:
            # 1. Price above daily EMA50 (bullish bias)
            # 2. Price ratio > 1.005 (0.5% above weekly EMA - weak uptrend)
            # 3. Volume confirmation
            if close_val > ema50_val and ratio_val > 1.005 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Price below daily EMA50 (bearish bias)
            # 2. Price ratio < 0.995 (0.5% below weekly EMA - weak downtrend)
            # 3. Volume confirmation
            elif close_val < ema50_val and ratio_val < 0.995 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below daily EMA50 or ratio breaks down
            if close_val < ema50_val or ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above daily EMA50 or ratio breaks up
            if close_val > ema50_val or ratio_val > 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals