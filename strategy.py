#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# Uses weekly EMA(21) for trend direction, breaks above/below 20-period Donchian channels for entries
# Only takes breakouts when volume > 1.5x 20-period average for confirmation
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 30-100 total trades over 4 years (7-25/year) to balance edge and fee drag
# Works in both bull/bear: weekly EMA filter ensures we trade with the higher timeframe trend

name = "1d_1w_donchian_volume_ema_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 20-period Donchian channels on 1d
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-20:i])
            lowest_low[i] = np.min(low[i-20:i])
    
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
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and weekly EMA trend filter
            if volume_confirm:
                # Long breakout: price closes above Donchian upper band AND price > weekly EMA
                if close[i] > highest_high[i] and close[i] > ema_21_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian lower band AND price < weekly EMA
                elif close[i] < lowest_low[i] and close[i] < ema_21_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals