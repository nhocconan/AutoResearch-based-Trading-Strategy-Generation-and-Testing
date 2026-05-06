#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla H4/L4 breakout with 1w Supertrend trend filter and volume confirmation
# Long when price breaks above 1d Camarilla H4 AND 1w Supertrend is bullish AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below 1d Camarilla L4 AND 1w Supertrend is bearish AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price crosses back through the 1d Camarilla midpoint (H4/L4 average)
# Uses discrete sizing 0.30 to balance return and risk
# Target: 100-250 total trades over 4 years (25-62/year) for 4h timeframe
# 1d Camarilla H4/L4 provides strong breakout levels that reduce whipsaw vs H3/L3
# 1w Supertrend ensures we trade with the dominant weekly trend with built-in ATR stop
# Volume confirmation (2.0x) validates breakout strength while limiting overtrading

name = "4h_1dCamarillaH4L4_1wSupertrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed 1d bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H4, L4, midpoint)
    # Camarilla: H4 = close + 1.5*(high-low)*1.1/2, L4 = close - 1.5*(high-low)*1.1/2
    high_low_1d = high_1d - low_1d
    camarilla_h4_1d = close_1d + 1.5 * high_low_1d * 1.1 / 2.0
    camarilla_l4_1d = close_1d - 1.5 * high_low_1d * 1.1 / 2.0
    camarilla_mid_1d = (camarilla_h4_1d + camarilla_l4_1d) / 2.0
    
    # Align 1d Camarilla to 4h timeframe (wait for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (ATR=10, mult=3.0)
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2.0
    upper_band = hl2 + (3.0 * atr_1w)
    lower_band = hl2 - (3.0 * atr_1w)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        # Calculate upper and lower bands
        upper_band[i] = hl2[i] + (3.0 * atr_1w[i])
        lower_band[i] = hl2[i] - (3.0 * atr_1w[i])
        
        # Adjust bands based on previous close
        if close_1w[i-1] > supertrend[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        # Determine Supertrend and direction
        if close_1w[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    # Align 1w Supertrend direction to 4h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(supertrend_direction_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla H4, 1w Supertrend bullish (direction=1), volume confirmation, in session
            if (close[i] > camarilla_h4_aligned[i] and 
                supertrend_direction_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Camarilla L4, 1w Supertrend bearish (direction=-1), volume confirmation, in session
            elif (close[i] < camarilla_l4_aligned[i] and 
                  supertrend_direction_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back above 1d Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals