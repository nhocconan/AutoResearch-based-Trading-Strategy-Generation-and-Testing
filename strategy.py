#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d OHLC-based pivot points (classic floor trader pivots) 
# combined with 60-period EMA trend filter on 6h and volume confirmation. 
# Pivots provide objective support/resistance levels derived from prior day's action. 
# Long when price crosses above R1 with EMA60 uptrend and volume > 1.5x average. 
# Short when price crosses below S1 with EMA60 downtrend and volume confirmation. 
# Exit when price touches opposite pivot level (S1 for long, R1 for short) or EMA crosses opposite direction. 
# Uses daily pivots which adapt to volatility and work in both trending and ranging markets. 
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate classic pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Load 6h data ONCE for EMA60 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    ema_6h = pd.Series(close_6h).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Align indicators to lower timeframe (primary timeframe is 6h, so no alignment needed for 6h EMA)
    # But we need to align daily pivots to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # For 6h EMA, since primary timeframe is 6h, we can use directly but need to align index
    # Actually, we're already on 6h timeframe, so no alignment needed
    ema_6h_aligned = ema_6h  # Already on correct timeframe
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, 20)  # Need EMA60 and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(ema_6h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: EMA60 slope (simplified: current vs previous)
        ema_now = ema_6h_aligned[i]
        ema_prev = ema_6h_aligned[i-1] if i > 0 else ema_now
        ema_uptrend = ema_now > ema_prev
        ema_downtrend = ema_now < ema_prev
        
        if position == 0:
            # Look for pivot breakouts
            # Long: price crosses above R1 with EMA uptrend and volume
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                ema_uptrend and volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below S1 with EMA downtrend and volume
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  ema_downtrend and volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches S1 or EMA turns down
            if (close[i] <= s1_aligned[i] or not ema_uptrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches R1 or EMA turns up
            if (close[i] >= r1_aligned[i] or not ema_downtrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_PivotPoints_EMA60_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0