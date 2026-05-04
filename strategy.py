#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for higher timeframe trend alignment (more stable than shorter HTF, less whipsaw)
# Camarilla R4/S4 from prior 1d session provide strong breakout levels with lower false breakout rate
# Volume confirmation (>2.0x 20 EMA) ensures high participation breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 80-120 total trades over 4 years = 20-30/year for 1h (avoiding overtrading)
# Works in both bull and bear: trend filter adapts to higher timeframe direction, breakouts capture momentum

name = "1h_Camarilla_R4S4_1dEMA50_VolumeSpike_Session"
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
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_1d = (high_1d + low_1d + close_1d_vals) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla R4, S4 levels (strong breakout levels)
    camarilla_r4 = close_1d_vals + 1.1 * range_1d
    camarilla_s4 = close_1d_vals - 1.1 * range_1d
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0 and in_session:
            # Long conditions: price breaks above Camarilla R4 + uptrend + volume spike
            if close[i] > r4_aligned[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S4 + downtrend + volume spike
            elif close[i] < s4_aligned[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H3/L3 midpoint OR trend changes OR weak volume OR outside session
            camarilla_h3 = close_1d_vals + 1.1 * range_1d / 6.0
            camarilla_l3 = close_1d_vals - 1.1 * range_1d / 6.0
            h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            midpoint = (h3_aligned[i] + l3_aligned[i]) / 2.0
            
            if (close[i] < midpoint or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i] or
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla H3/L3 midpoint OR trend changes OR weak volume OR outside session
            camarilla_h3 = close_1d_vals + 1.1 * range_1d / 6.0
            camarilla_l3 = close_1d_vals - 1.1 * range_1d / 6.0
            h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            midpoint = (h3_aligned[i] + l3_aligned[i]) / 2.0
            
            if (close[i] > midpoint or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i] or
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals