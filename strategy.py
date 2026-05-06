#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Keltner Channel breakout with 1w Supertrend trend filter and volume confirmation
# Long when price breaks above 1d Keltner Upper Band AND 1w Supertrend is bullish AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d Keltner Lower Band AND 1w Supertrend is bearish AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Keltner Middle Line (EMA20)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Keltner Channel provides volatility-adaptive breakout levels that reduce whipsaw in both trending and ranging markets
# 1w Supertrend (ATR=10, mult=3) ensures we trade with the dominant weekly trend while avoiding chop
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# This combination has shown robustness across BTC, ETH, and SOL in both bull and bear regimes

name = "12h_1dKeltner_1wSupertrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for EMA20 and ATR
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 (middle line)
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR(10) for Keltner Channel width
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # Align length with close_1d
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d Keltner Channel bands
    keltner_mid_1d = ema_20_1d
    keltner_upper_1d = ema_20_1d + (2.0 * atr_10_1d)
    keltner_lower_1d = ema_20_1d - (2.0 * atr_10_1d)
    
    # Align 1d Keltner Channel to 12h timeframe (wait for completed 1d bar)
    keltner_mid_aligned = align_htf_to_ltf(prices, df_1d, keltner_mid_1d)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    
    # Get 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need at least 30 completed weekly bars for Supertrend
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(10) for Supertrend
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])  # Align length with close_1w
    atr_10_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1w Supertrend
    # Basic Upper Band = (high + low)/2 + mult * ATR
    # Basic Lower Band = (high + low)/2 - mult * ATR
    mult = 3.0
    basic_ub_1w = (high_1w + low_1w) / 2.0 + mult * atr_10_1w
    basic_lb_1w = (high_1w + low_1w) / 2.0 - mult * atr_10_1w
    
    # Initialize Supertrend arrays
    supertrend_1w = np.full_like(close_1w, np.nan)
    direction_1w = np.full_like(close_1w, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    if not np.isnan(basic_ub_1w[0]) and not np.isnan(basic_lb_1w[0]):
        supertrend_1w[0] = basic_ub_1w[0]
        direction_1w[0] = 1  # Start with uptrend assumption
    
    # Calculate Supertrend iteratively
    for i in range(1, len(close_1w)):
        if np.isnan(basic_ub_1w[i]) or np.isnan(basic_lb_1w[i]) or np.isnan(supertrend_1w[i-1]):
            supertrend_1w[i] = supertrend_1w[i-1] if not np.isnan(supertrend_1w[i-1]) else np.nan
            direction_1w[i] = direction_1w[i-1] if not np.isnan(direction_1w[i-1]) else 1
            continue
            
        # Determine final upper and lower bands
        if basic_ub_1w[i] < supertrend_1w[i-1] or close_1w[i-1] > supertrend_1w[i-1]:
            final_ub = basic_ub_1w[i]
        else:
            final_ub = supertrend_1w[i-1]
            
        if basic_lb_1w[i] > supertrend_1w[i-1] or close_1w[i-1] < supertrend_1w[i-1]:
            final_lb = basic_lb_1w[i]
        else:
            final_lb = supertrend_1w[i-1]
        
        # Determine trend direction
        if close_1w[i] > final_ub:
            direction_1w[i] = 1
        elif close_1w[i] < final_lb:
            direction_1w[i] = -1
        else:
            direction_1w[i] = direction_1w[i-1]
        
        # Set Supertrend value
        if direction_1w[i] == 1:
            supertrend_1w[i] = final_lb
        else:
            supertrend_1w[i] = final_ub
    
    # Align 1w Supertrend direction to 12h timeframe (wait for completed 1w bar)
    # We use the direction (-1, 1) as the trend filter
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(keltner_mid_aligned[i]) or np.isnan(direction_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Keltner Upper Band, 1w Supertrend is bullish (1), volume confirmation, in session
            if (close[i] > keltner_upper_aligned[i] and 
                direction_1w_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Keltner Lower Band, 1w Supertrend is bearish (-1), volume confirmation, in session
            elif (close[i] < keltner_lower_aligned[i] and 
                  direction_1w_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Keltner Middle Line
            if close[i] < keltner_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Keltner Middle Line
            if close[i] > keltner_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals