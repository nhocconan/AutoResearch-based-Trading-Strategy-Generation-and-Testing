#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h fractal breakout with 1d trend and volume confirmation
# Uses Williams Fractals for swing high/low detection, 1d EMA for trend filter, and volume spike for confirmation
# Designed to work in both bull and bear markets by requiring strong trend alignment and volume confirmation
# Target: 12-37 trades/year, focused on high-probability breakouts with confirmation
name = "12h_fractal_breakout_1d_trend_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume context (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA for trend filter (50-period)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume SMA for volume context (20-period)
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Fractals on 12h data
    # Bearish fractal: high[i] is highest among 5 bars (i-2, i-1, i, i+1, i+2)
    # Bullish fractal: low[i] is lowest among 5 bars (i-2, i-1, i, i+1, i+2)
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i] >= high[i-1] and high[i] >= high[i-2] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = True
        if (low[i] <= low[i-1] and low[i] <= low[i-2] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d[i]) or np.isnan(volume_1d[i]) or 
            np.isnan(vol_sma_1d[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 12h bar
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)[i]
        
        # Trend filter: price above/below 50 EMA on 1d
        uptrend = close[i] > ema_1d_aligned
        downtrend = close[i] < ema_1d_aligned
        
        # Volume filter: current volume above 2.0x 1d average volume
        volume_filter = volume[i] > (vol_sma_1d_aligned * 2.0)
        
        if position == 1:  # Long position
            # Exit: bullish fractal broken OR trend reversal
            if bullish_fractal[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bearish fractal broken OR trend reversal
            if bearish_fractal[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bullish fractal forms + uptrend + volume filter
            if bullish_fractal[i] and uptrend and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short: bearish fractal forms + downtrend + volume filter
            elif bearish_fractal[i] and downtrend and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals