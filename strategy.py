#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot + volume confirmation + 12h trend filter
# Camarilla levels from 1d: R3/S3 for mean reversion, R4/S4 for breakout continuation
# 12h EMA(20) trend filter ensures alignment with higher timeframe momentum
# Volume confirmation filters out low-liquidity false signals
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in bull/bear via mean reversion in range, breakout in trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.zeros(len(close_12h))
    ema_multiplier = 2 / (20 + 1)
    ema_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * ema_multiplier + ema_12h[i-1]
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        trend = ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # Calculate Camarilla levels from previous 1d bar
        if i >= 1:
            # Get previous day's OHLC (1d bar that closed before current 6h bar)
            prev_day_idx = min(i // 4, len(df_1d) - 1)  # 4x 6h bars per day
            if prev_day_idx >= 1:
                phigh = df_1d['high'].iloc[prev_day_idx - 1]
                plow = df_1d['low'].iloc[prev_day_idx - 1]
                pclose = df_1d['close'].iloc[prev_day_idx - 1]
                
                # Camarilla levels
                range_val = phigh - plow
                if range_val > 0:
                    # Resistance levels
                    r3 = pclose + (range_val * 1.1 / 2)
                    r4 = pclose + (range_val * 1.1)
                    # Support levels
                    s3 = pclose - (range_val * 1.1 / 2)
                    s4 = pclose - (range_val * 1.1)
                    
                    if position == 0:
                        # Long setup: mean reversion at S3 OR breakout above R4 with trend
                        if (price <= s3 and trend > pclose) or (price >= r4 and trend > pclose):
                            if volume_confirm:
                                position = 1
                                signals[i] = position_size
                        # Short setup: mean reversion at R3 OR breakdown below S4 with trend
                        elif (price >= r3 and trend < pclose) or (price <= s4 and trend < pclose):
                            if volume_confirm:
                                position = -1
                                signals[i] = -position_size
                    elif position == 1:
                        # Exit long: price reaches R3 (mean reversion target) or breaks R4 without trend
                        if price >= r3 or (price >= r4 and trend < pclose):
                            position = 0
                            signals[i] = 0.0
                    elif position == -1:
                        # Exit short: price reaches S3 (mean reversion target) or breaks S4 without trend
                        if price <= s3 or (price <= s4 and trend > pclose):
                            position = 0
                            signals[i] = 0.0
    
    return signals

name = "6h_12h_Camarilla_Pivot_Volume_Trend"
timeframe = "6h"
leverage = 1.0