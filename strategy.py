#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Donchian breakouts capture momentum; 12h EMA50 ensures trend alignment (avoids counter-trend trades);
# volume confirms breakout strength. Works in bull/bear by only trading with the 12h trend.
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 (trend filter)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 12h EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine trend based on 12h EMA50
        bullish_trend = curr_close > curr_ema_50_12h
        bearish_trend = curr_close < curr_ema_50_12h
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band + bullish trend + volume
            if curr_high > curr_highest_high and bullish_trend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + bearish trend + volume
            elif curr_low < curr_lowest_low and bearish_trend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below Donchian lower band OR trend turns bearish
            if curr_low < curr_lowest_low or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above Donchian upper band OR trend turns bullish
            if curr_high > curr_highest_high or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals