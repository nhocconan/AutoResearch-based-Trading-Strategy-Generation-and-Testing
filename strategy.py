#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA trend + volume confirmation
# Uses Williams %R(14) for overbought/oversold signals on 6h timeframe
# Only takes trades when 12h EMA(50) confirms trend direction (EMA up for longs, EMA down for shorts)
# Volume confirmation: current volume > 1.3x 20-period average to avoid low-volume false signals
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag
# Works in both bull/bear: Williams %R captures mean reversion in ranging markets,
# while 12h EMA filter ensures we only trade with the higher timeframe trend

name = "6h_12h_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14)
    williams_r = np.full(len(df_6h), np.nan)
    highest_high = np.full(len(df_6h), np.nan)
    lowest_low = np.full(len(df_6h), np.nan)
    
    for i in range(len(df_6h)):
        if i < 14:
            williams_r[i] = np.nan
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            # Lookback window excluding current bar to avoid look-ahead
            highest_high[i] = np.max(df_6h['high'].iloc[i-14:i])
            lowest_low[i] = np.min(df_6h['low'].iloc[i-14:i])
            if highest_high[i] == lowest_low[i]:
                williams_r[i] = -50  # Avoid division by zero
            else:
                williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        multiplier = 2 / (50 + 1)
        ema_12h[49] = np.mean(close_12h[:50])  # SMA for first value
        for i in range(50, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Align 6h Williams %R to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Align 12h EMA to 6h timeframe
    ema_12h_6h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_6h[i]) or 
            np.isnan(ema_12h_6h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R > -20 (overbought) OR volume confirmation fails
            if williams_r_6h[i] > -20 or not volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -80 (oversold) OR volume confirmation fails
            if williams_r_6h[i] < -80 or not volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extreme with 12h EMA trend filter and volume confirmation
            if volume_confirm:
                # Long entry: Williams %R < -80 (oversold) AND 12h EMA rising
                if williams_r_6h[i] < -80 and ema_12h_6h[i] > ema_12h_6h[max(0, i-1)]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought) AND 12h EMA falling
                elif williams_r_6h[i] > -20 and ema_12h_6h[i] < ema_12h_6h[max(0, i-1)]:
                    position = -1
                    signals[i] = -0.25
    
    return signals