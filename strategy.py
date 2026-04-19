#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour EMA trend filter and volume confirmation.
# Donchian channels identify price extremes and breakout opportunities.
# We trade breakouts in the direction of the 12h trend (EMA50) with volume confirmation.
# Works in bull/bear markets: avoids false breakouts in ranging markets, captures true breakouts.
# Target: 20-50 trades/year per symbol.
name = "4h_Donchian20_EMA50_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian Channel (20 period) on 4h
    dc_period = 20
    upper_channel = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_channel = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Align 12h EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_period, 50, 20)  # Ensure Donchian, EMA50, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper  # Price breaks above upper channel
        bearish_breakout = price < lower  # Price breaks below lower channel
        
        if position == 0:
            # Look for entry after Donchian breakout, in direction of 12h trend
            if bullish_breakout and (price > ema_50_val) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and (price < ema_50_val) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to middle of channel (mean reversion)
            middle_channel = (upper + lower) / 2
            if price < middle_channel:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle of channel
            middle_channel = (upper + lower) / 2
            if price > middle_channel:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals