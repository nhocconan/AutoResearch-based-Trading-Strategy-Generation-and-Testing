#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla H3 AND price > 4h EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla L3 AND price < 4h EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses Camarilla H4/L4 (extreme) OR volume < avg_volume(20)
# Uses discrete sizing 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Session filter: 08-20 UTC to reduce noise trades
# Camarilla levels from 1d provide robust daily support/resistance; 4h EMA50 filters intermediate trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 for previous day calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's high, low, close
    # H3 = close + 1.1 * (high - low) / 4  (R3)
    # L3 = close - 1.1 * (high - low) / 4  (S3)
    # H4 = close + 1.1 * (high - low) / 2  (R4)
    # L4 = close - 1.1 * (high - low) / 2  (S4)
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d / 4  # R3
    camarilla_l3 = close_1d - 1.1 * range_1d / 4  # S3
    camarilla_h4 = close_1d + 1.1 * range_1d / 2  # R4
    camarilla_l4 = close_1d - 1.1 * range_1d / 2  # S4
    
    # Align Camarilla levels to 1h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla H3, above 4h EMA50, volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla L3, below 4h EMA50, volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla L4 (extreme support) OR volume drops below average
            if close[i] < camarilla_l4_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price crosses above Camarilla H4 (extreme resistance) OR volume drops below average
            if close[i] > camarilla_h4_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals