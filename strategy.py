#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 Breakout with 1d EMA50 Trend Filter and Volume Spike
- Camarilla R3/S3 levels act as strong intraday support/resistance on 1d timeframe
- Breakouts beyond R3/S3 with volume confirmation indicate strong momentum
- 1d EMA(50) filter ensures trades align with daily trend to avoid counter-trend whipsaws
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Volume spike (>2.0x 20-period average) filters weak breakouts
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
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need enough data for pivot and EMA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(Pivot - L)
    # S3 = L - 2*(H - Pivot)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = high_1d + 2.0 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2.0 * (high_1d - pivot_1d)
    
    # Align Camarilla levels to 6h timeframe (no extra delay needed for pivot points)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above R3 with volume and uptrend
        # Short: price breaks below S3 with volume and downtrend
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above R3, uptrend, volume spike
            long_signal = (price_above_r3 and 
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below S3, downtrend, volume spike
            short_signal = (price_below_s3 and 
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite level break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S3 or trend turns down
                if (price_below_s3 or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 or trend turns up
                if (price_above_r3 or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0