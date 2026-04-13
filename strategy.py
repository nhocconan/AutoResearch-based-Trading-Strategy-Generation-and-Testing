# 12h_1w_Camarilla_Breakout_Volume_Trend
# Hypothesis: 12-hour Camarilla breakout with 1-week trend filter and volume confirmation.
# Weekly EMA(21) filters long-term direction, reducing counter-trend whipsaw.
# Breakouts above weekly R4 or below S4 with volume confirmation capture strong momentum.
# Designed for 12h timeframe to achieve 12-37 trades/year (50-150 total over 4 years).
# Works in bull markets by riding breakouts with trend, and in bear by avoiding counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (21 + 1)
    ema_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w[i] = (close_1w[i] - ema_1w[i-1]) * ema_multiplier + ema_1w[i-1]
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla levels from previous week
    # Using weekly high, low, close from previous completed week
    phigh = np.roll(df_1w['high'].values, 1)
    plow = np.roll(df_1w['low'].values, 1)
    pclose = np.roll(df_1w['close'].values, 1)
    phigh[0] = plow[0] = pclose[0] = np.nan  # First week has no previous
    
    # Camarilla formulas
    range_ = phigh - plow
    r4 = pclose + range_ * 1.1 / 2
    r3 = pclose + range_ * 1.1 / 4
    s3 = pclose - range_ * 1.1 / 4
    s4 = pclose - range_ * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate average volume (2-period = 1 day) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(2, n):
        avg_volume[i] = np.mean(volume[i-2:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(2, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long breakout: price > R4 + above weekly EMA + volume confirmation
            if (price > r4_aligned[i] and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short breakdown: price < S4 + below weekly EMA + volume confirmation
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

name = "12h_1w_Camarilla_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0