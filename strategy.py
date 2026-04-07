#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Uses daily Donchian channel (20) to identify breakouts in the direction of weekly trend
# - Long when price breaks above Donchian upper band AND weekly trend is up
# - Short when price breaks below Donchian lower band AND weekly trend is down
# - Weekly trend determined by EMA50 on weekly timeframe
# - Volume confirmation ensures breakout is supported by participation
# - Designed for low frequency (target: 10-25 trades/year) to minimize fee impact
# - Works in bull/bear via trend filter: only trade in direction of higher timeframe trend

name = "1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly trend filter (EMA50)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Daily Donchian channel (20)
    donchian_window = 20
    dc_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_50_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from weekly EMA
        uptrend = close[i] > ema_50_weekly_aligned[i]
        downtrend = close[i] < ema_50_weekly_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when price breaks below Donchian lower or trend changes
            if close[i] < dc_lower[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper or trend changes
            if close[i] > dc_upper[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long breakout: price above Donchian upper with uptrend and volume
            if close[i] > dc_upper[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price below Donchian lower with downtrend and volume
            elif close[i] < dc_lower[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals