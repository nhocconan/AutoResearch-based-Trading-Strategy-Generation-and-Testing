# 4h_Donchian_Breakout_Volume_Trend
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h trend filter (EMA34) for trend strength.
# Long when price breaks above 20-period high + volume > 1.5x average + EMA34 rising.
# Short when price breaks below 20-period low + volume > 1.5x average + EMA34 falling.
# Designed to capture trends with filtered entries, reducing false breakouts in ranging markets.
# Position size 0.25 balances return and drawdown control.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema34_12h = calculate_ema(close_12h, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Donchian channels (20-period) and average volume
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    avg_volume = np.full_like(volume, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
        avg_volume[i] = np.mean(volume[i-lookback+1:i+1])
    
    # Volume threshold: 1.5x average
    volume_threshold = avg_volume * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, lookback-1)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high + volume confirmation + EMA34 rising
            if (close[i] > highest_high[i] and 
                volume[i] > volume_threshold[i] and 
                ema34_12h_aligned[i] > ema34_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low + volume confirmation + EMA34 falling
            elif (close[i] < lowest_low[i] and 
                  volume[i] > volume_threshold[i] and 
                  ema34_12h_aligned[i] < ema34_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-period low OR EMA34 turns down
            if close[i] < lowest_low[i] or ema34_12h_aligned[i] < ema34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-period high OR EMA34 turns up
            if close[i] > highest_high[i] or ema34_12h_aligned[i] > ema34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals