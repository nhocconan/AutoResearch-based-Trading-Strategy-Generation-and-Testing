# 1D_Wilson_Wave_Oscillator_Trend_Filter
# Combines Wilson Wave Oscillator with trend filter and volume confirmation
# Wilson Wave Oscillator: %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
# Trend filter: 1w EMA200 on weekly chart for long-term direction
# Volume confirmation: Daily volume > 1.5x 20-day average
# Entry: WWO oversold (<20) in uptrend or overbought (>80) in downtrend + volume confirmation
# Exit: Opposite WWO level or trend reversal
# Designed for 1d timeframe to work in both bull and bear markets
# Target: 15-25 trades/year to stay within fee limits

#!/usr/bin/env python3
name = "1D_Wilson_Wave_Oscillator_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Wilson Wave Oscillator (14-period)
    wwo_period = 14
    highest_high = pd.Series(high).rolling(window=wwo_period, min_periods=wwo_period).max().values
    lowest_low = pd.Series(low).rolling(window=wwo_period, min_periods=wwo_period).min().values
    
    # Avoid division by zero
    range_ww = highest_high - lowest_low
    wwo_raw = np.where(range_ww != 0, (close - lowest_low) / range_ww * 100, 50)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day volume average
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align volume MA to daily timeframe (same timeframe, so direct use)
    vol_ma20_aligned = vol_ma20_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(200, wwo_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(wwo_raw[i]) or 
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 1w EMA200
        uptrend = close[i] > ema200_1w_aligned[i]
        # Downtrend: price below 1w EMA200
        downtrend = close[i] < ema200_1w_aligned[i]
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > vol_ma20_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: WWO oversold (<20) in uptrend + volume confirmation
            if uptrend and wwo_raw[i] < 20 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: WWO overbought (>80) in downtrend + volume confirmation
            elif downtrend and wwo_raw[i] > 80 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: WWO overbought (>80) or trend turns down
            if wwo_raw[i] > 80 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WWO oversold (<20) or trend turns up
            if wwo_raw[i] < 20 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals