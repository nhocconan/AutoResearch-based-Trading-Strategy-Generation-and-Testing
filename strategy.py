#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# Long when price breaks above R4 with volume confirmation AND 1d EMA34 > EMA34 previous (uptrend)
# Short when price breaks below S4 with volume confirmation AND 1d EMA34 < EMA34 previous (downtrend)
# Exit when price reverts to R3/S3 levels or opposite extreme
# Uses 6h timeframe for better trade frequency control vs 4h, with 1d/1w HTF for structure
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Camarilla levels provide mathematically derived support/resistance from prior day's range
# Breakout at R4/S4 captures strong momentum; reversion to R3/S3 provides defined exits
# Volume confirmation (2.0x) validates breakout strength while limiting false signals
# 1d EMA34 trend filter ensures we trade with dominant daily trend to avoid whipsaws
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns)

name = "6h_Camarilla_R4S4_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed daily bars for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Based on previous day's high, low, close
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll, handle it
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Camarilla calculations
    range_1d = prev_high_1d - prev_low_1d
    # Avoid division by zero
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    # Camarilla levels
    R3 = prev_close_1d + range_1d * 1.1 / 4
    R4 = prev_close_1d + range_1d * 1.1 / 2
    S3 = prev_close_1d - range_1d * 1.1 / 4
    S4 = prev_close_1d - range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 with volume confirmation AND 1d EMA34 > EMA34 previous (uptrend)
            if (close[i] > R4_aligned[i] and 
                volume_confirm[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume confirmation AND 1d EMA34 < EMA34 previous (downtrend)
            elif (close[i] < S4_aligned[i] and 
                  volume_confirm[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to R3 level or breaks above R4 strongly (trailing)
            if close[i] <= R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to S3 level or breaks below S4 strongly (trailing)
            if close[i] >= S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals