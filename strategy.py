#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 12h as primary timeframe for low trade frequency (target: 50-150 total trades over 4 years)
# 1w EMA50 provides strong trend filter to avoid counter-trend trades in ranging markets
# Donchian breakout captures momentum with clear entry/exit levels
# Volume confirmation (1.5x 24-period average) ensures institutional participation
# Designed for minimal trades to overcome fee drag in bear markets (2025+)
# Works in bull markets via trend-aligned breakouts, avoids false signals in chop via trend filter

name = "12h_Donchian20_1wEMA50_Volume_Confirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for volume average (more stable than 12h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume moving average (24-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=24, min_periods=24).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 12h Donchian channels (20-period)
    # Highest high over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low over 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper band + price > 1w EMA50 + volume > 1.5x 1d avg
            if close[i] > highest_high[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (vol_ma_1d_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band + price < 1w EMA50 + volume > 1.5x 1d avg
            elif close[i] < lowest_low[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (vol_ma_1d_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower band (mean reversion) or trend reversal
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper band (mean reversion) or trend reversal
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals