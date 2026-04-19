#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1-week trend filter and volume confirmation.
# Long when price > Alligator Jaw, weekly trend up, volume > 1.5x average.
# Short when price < Alligator Jaw, weekly trend down, volume > 1.5x average.
# Designed for 12h timeframe to capture multi-day trends with reduced whipsaw.
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years).
name = "12h_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    # We'll use Jaw as the main reference line
    def smma(source, length):
        result = np.full_like(source, np.nan)
        if len(source) < length:
            return result
        # First value is simple average
        result[length-1] = np.mean(source[0:length])
        # Subsequent values: SMMA = (PREV * (LENGTH-1) + CURRENT) / LENGTH
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    jaw = smma(close, 13)
    jaw_shifted = np.roll(jaw, 8)  # Shift 8 bars forward
    jaw_shifted[:8] = np.nan  # Fill shifted portion with NaN
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA13 trend
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align weekly EMA13 to 12h
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13+8)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_shifted[i]) or np.isnan(ema_13_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_shifted[i]
        weekly_trend = ema_13_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Determine if weekly trend is up or down
        weekly_up = weekly_trend > 0  # Using positive slope approximation
        weekly_down = weekly_trend < 0
        
        if position == 0:
            # Enter long if price above Jaw, weekly trend up, and volume confirmation
            if price > jaw_val and weekly_up and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price below Jaw, weekly trend down, and volume confirmation
            elif price < jaw_val and weekly_down and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below Jaw or weekly trend turns down
            if price < jaw_val or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above Jaw or weekly trend turns up
            if price > jaw_val or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals