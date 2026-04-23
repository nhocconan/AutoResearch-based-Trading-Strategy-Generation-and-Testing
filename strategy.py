#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1w EMA50 Trend Filter and Volume Spike
- Long when Williams %R < -90 (extreme oversold) AND close > 1w EMA50 AND volume > 2.5x 24-period average
- Short when Williams %R > -10 (extreme overbought) AND close < 1w EMA50 AND volume > 2.5x 24-period average
- Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
- Uses 1w EMA50 for HTF trend alignment to ensure we only trade with the weekly trend
- Volume spike threshold set to 2.5x to reduce false signals and control trade frequency
- Designed for both bull and bear markets: trend filter prevents counter-trend entries, Williams %R captures reversals in extended moves
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
"""

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
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams %R calculation (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 2.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 14)  # Need 50 for EMA50, 24 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme conditions
        williams_oversold = williams_r[i] < -90  # Extreme oversold
        williams_overbought = williams_r[i] > -10  # Extreme overbought
        williams_exit_long = williams_r[i] > -50   # Exit long when crosses above -50
        williams_exit_short = williams_r[i] < -50  # Exit short when crosses below -50
        
        # Trend filter (using 1w EMA50)
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation (stricter threshold)
        volume_ok = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R extreme oversold + uptrend + volume confirmation
            if williams_oversold and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extreme overbought + downtrend + volume confirmation
            elif williams_overbought and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if williams_exit_long:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses below -50
                if williams_exit_short:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0