#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(40) breakout with 12-hour trend filter and volume confirmation
# Uses longer Donchian period (40) to reduce trade frequency and avoid whipsaws.
# 12-hour trend filter aligns with medium-term direction to avoid counter-trend trades.
# Volume confirmation ensures institutional participation in breakouts.
# Designed for very low frequency: target 15-30 trades/year to minimize fee drag.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

name = "4h_donchian40_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(20) for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_12h = close_12h.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (40-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=40, min_periods=40).max().values
    donchian_low = low_series.rolling(window=40, min_periods=40).min().values
    
    # Volume confirmation (40-period average)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # 12-hour trend filter: price above/below EMA(20)
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: opposite Donchian break
        exit_long = close[i] < donchian_low[i-1]
        exit_short = close[i] > donchian_high[i-1]
        
        if position == 1:  # Long position
            # Exit on breakdown or trend reversal
            if exit_long or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on breakout or trend reversal
            if exit_short or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: upward breakout + uptrend + volume confirmation
            if breakout_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: downward breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals