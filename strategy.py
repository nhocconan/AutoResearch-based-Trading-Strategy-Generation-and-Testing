#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 to confirm trend direction and avoid whipsaws in both bull/bear markets
# Camarilla H3/L3 from prior 1d session provide institutional breakout levels
# Volume confirmation (>1.8x 20 EMA) filters low-participation false breakouts
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in both bull and bear: EMA50 ensures we only trade with the trend, Camarilla provides precise entry/exit levels.

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data for Camarilla levels (daily session)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_1d = (high_1d + low_1d + close_1d_vals) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla H3, L3 levels (most significant for breakouts)
    camarilla_h3 = close_1d_vals + 1.1 * range_1d / 4.0
    camarilla_l3 = close_1d_vals - 1.1 * range_1d / 4.0
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla H3 + price above 4h EMA50 + volume spike
            if close[i] > h3_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla L3 + price below 4h EMA50 + volume spike
            elif close[i] < l3_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H4/L4 midpoint OR price crosses below 4h EMA50
            camarilla_h4 = close_1d_vals + 1.1 * range_1d / 4.0
            camarilla_l4 = close_1d_vals - 1.1 * range_1d / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if close[i] < midpoint or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla H4/L4 midpoint OR price crosses above 4h EMA50
            camarilla_h4 = close_1d_vals + 1.1 * range_1d / 4.0
            camarilla_l4 = close_1d_vals - 1.1 * range_1d / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if close[i] > midpoint or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals