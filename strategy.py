#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with volume confirmation and 1w trend filter
    # Long: Close breaks above R4 + volume > 1.5x 20-period average + price > 1w EMA200 (uptrend)
    # Short: Close breaks below S4 + volume > 1.5x 20-period average + price < 1w EMA200 (downtrend)
    # Exit: Close retreats to R3/S3 levels or opposite pivot break
    # Uses 1w EMA200 for major trend alignment to avoid counter-trend trades
    # Camarilla R4/S4 represent strong breakout levels; R3/S3 represent pullback targets
    # Volume filter ensures breakouts have conviction
    # Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1w data for EMA200 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 6h Camarilla levels (based on previous day's range)
    # Camarilla: R4 = Close + 1.1*(High-Low)*1.1/2, S4 = Close - 1.1*(High-Low)*1.1/2
    # R3 = Close + 1.1*(High-Low)*1.1/4, S3 = Close - 1.1*(High-Low)*1.1/4
    # Using 6h bar's range as proxy for previous period (common in intraday)
    range_6h = high_6h - low_6h
    camarilla_multiplier = 1.1 * 1.1 / 2  # 1.21/2 = 0.605 for R4/S4
    camarilla_multiplier_half = 1.1 * 1.1 / 4  # 1.21/4 = 0.3025 for R3/S3
    
    r4 = close_6h + range_6h * camarilla_multiplier
    s4 = close_6h - range_6h * camarilla_multiplier
    r3 = close_6h + range_6h * camarilla_multiplier_half
    s3 = close_6h - range_6h * camarilla_multiplier_half
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for volume MA
        # Skip if data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(r3[i]) or np.isnan(s3[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > r4[i] and volume_filter[i]
        breakout_short = close[i] < s4[i] and volume_filter[i]
        
        # Trend filter from 1w EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Exit conditions: retreat to R3/S3 or opposite breakout
        exit_long = position == 1 and (close[i] < r3[i] or breakout_short)
        exit_short = position == -1 and (close[i] > s3[i] or breakout_long)
        
        # Entry conditions
        long_entry = breakout_long and uptrend and position != 1
        short_entry = breakout_short and downtrend and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_camarilla_breakout_volume_filter_v1"
timeframe = "6h"
leverage = 1.0