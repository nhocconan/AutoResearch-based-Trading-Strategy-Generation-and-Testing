#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla H4/L4 breakout with 1w Supertrend trend filter and volume confirmation
# Long when price breaks above 1d Camarilla H4 AND 1w Supertrend is bullish AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 1d Camarilla L4 AND 1w Supertrend is bearish AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price crosses back through the 1d Camarilla midpoint (H4/L4 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Camarilla H4/L4 provides strong breakout levels that reduce whipsaw
# 1w Supertrend ensures we trade with the dominant weekly trend
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# Supertrend calculation: ATR(10) * 3, upper/lower bands, trend direction

name = "4h_1dCamarillaH4L4_1wSupertrend_Trend_VolumeConfirm"
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
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    high_low_1d = high_1d - low_1d
    camarilla_h4_1d = close_1d + 1.1 * high_low_1d * 1.1 / 2.0
    camarilla_l4_1d = close_1d - 1.1 * high_low_1d * 1.1 / 2.0
    camarilla_mid_1d = (camarilla_h4_1d + camarilla_l4_1d) / 2.0
    
    # Align 1d Camarilla to 4h timeframe (wait for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for ATR
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr1.iloc[0] = high_1w[0] - low_1w[0]  # First period
    tr2.iloc[0] = np.abs(high_1w[0] - close_1w[0])
    tr3.iloc[0] = np.abs(low_1w[0] - close_1w[0])
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR
    atr_1w = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (multiplier * atr_1w)
    lower_band = hl2 - (multiplier * atr_1w)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[atr_period-1] = upper_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close_1w)):
        if close_1w[i] <= supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = 1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align 1w Supertrend to 4h timeframe (wait for completed 1w bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
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
            # Long: price breaks above 1d Camarilla H4, 1w Supertrend is bullish (direction=1), volume confirmation, in session
            if (close[i] > camarilla_h4_aligned[i] and 
                supertrend_direction_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla L4, 1w Supertrend is bearish (direction=-1), volume confirmation, in session
            elif (close[i] < camarilla_l4_aligned[i] and 
                  supertrend_direction_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals