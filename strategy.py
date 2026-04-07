#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 20-period Donchian breakout with weekly EMA filter and volume confirmation
# Buy when price breaks above upper Donchian channel with price above weekly EMA and volume above average
# Sell when price breaks below lower Donchian channel with price below weekly EMA and volume above average
# Exit when price returns to the Donchian midline or on opposite breakout
# Designed for low frequency (7-25 trades/year) to minimize fee drag on 1d timeframe
# Works in bull markets via breakout continuation and bear markets via mean reversion at channel extremes

name = "1d_donchian20_weekly_ema_volume_v4"
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
    
    # Get weekly data for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Upper channel = highest high of past 20 days
    # Lower channel = lowest low of past 20 days
    # Middle channel = average of upper and lower
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (highest_high + lowest_low) / 2
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Donchian levels
        upper = highest_high[i]
        lower = lowest_low[i]
        middle = donchian_middle[i]
        
        # EMA filter
        price_above_ema = close[i] > ema_1w_aligned[i]
        price_below_ema = close[i] < ema_1w_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price returns to middle line or breaks below lower channel
            if close[i] <= middle or close[i] < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price returns to middle line or breaks above upper channel
            if close[i] >= middle or close[i] > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long on break above upper channel with volume confirmation and price above weekly EMA
            if close[i] > upper and vol_confirm and price_above_ema:
                position = 1
                signals[i] = 0.25
            # Enter short on break below lower channel with volume confirmation and price below weekly EMA
            elif close[i] < lower and vol_confirm and price_below_ema:
                position = -1
                signals[i] = -0.25
    
    return signals