#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with 1d trend filter and volume confirmation.
# Long when price breaks above R4 (strong breakout) in 1d uptrend with volume > 1.5x 20-period average.
# Short when price breaks below S4 (strong breakdown) in 1d downtrend with volume confirmation.
# Exit when price returns to the 12h pivot (central level) or trend weakens.
# Camarilla levels provide institutional reference points; breakouts beyond R4/S4 indicate strong momentum.
# Trend filter avoids counter-trend trades. Volume confirms institutional participation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for each 12h bar
    # Based on previous 12h bar's range
    range_12h = high_12h - low_12h
    # Pivot point (central)
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Resistance levels
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    r2_12h = close_12h + (range_12h * 1.1 / 6)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    r4_12h = close_12h + (range_12h * 1.1 / 2)
    # Support levels
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    s2_12h = close_12h - (range_12h * 1.1 / 6)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    s4_12h = close_12h - (range_12h * 1.1 / 2)
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA slope: current EMA - previous EMA
    ema_slope_1d = np.diff(ema_50_1d, prepend=np.nan)
    
    # Align indicators to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 50)  # Need volume MA and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r4_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: EMA slope positive for uptrend, negative for downtrend
        uptrend = ema_slope_1d_aligned[i] > 0
        downtrend = ema_slope_1d_aligned[i] < 0
        
        if position == 0:
            # Look for Camarilla breakouts
            # Long: price breaks above R4 AND uptrend
            if (close[i] > r4_12h_aligned[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below S4 AND downtrend
            elif (close[i] < s4_12h_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or trend weakens
            if (close[i] <= pivot_12h_aligned[i] or 
                ema_slope_1d_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot or trend weakens
            if (close[i] >= pivot_12h_aligned[i] or 
                ema_slope_1d_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0