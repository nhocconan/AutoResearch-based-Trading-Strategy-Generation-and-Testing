#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Camarilla levels provide clear support/resistance zones; breakouts above R4 or below S4
# indicate strong momentum when confirmed by 1d trend (EMA50) and volume spikes.
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.
# Target: 12-37 trades per year (50-150 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * ema_multiplier + ema_1d[i-1]
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous day
    # Using daily high, low, close from previous completed day
    phigh = np.roll(df_1d['high'].values, 1)
    plow = np.roll(df_1d['low'].values, 1)
    pclose = np.roll(df_1d['close'].values, 1)
    phigh[0] = plow[0] = pclose[0] = np.nan  # First day has no previous
    
    # Camarilla formulas
    range_ = phigh - plow
    r4 = pclose + range_ * 1.1 / 2
    r3 = pclose + range_ * 1.1 / 4
    s3 = pclose - range_ * 1.1 / 4
    s4 = pclose - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate average volume (24-period = 6 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long breakout: price > R4 + above daily EMA + volume confirmation
            if (price > r4_aligned[i] and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short breakdown: price < S4 + below daily EMA + volume confirmation
            elif (price < s4_aligned[i] and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < R3 (reversion to mean) or trend change
            if (price < r3_aligned[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > S3 (reversion to mean) or trend change
            if (price > s3_aligned[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0