#!/usr/bin/env python3
# 1d_price_channel_breakout_volume_v1
# Hypothesis: Daily price breaking above/below 20-day Donchian channels with volume
# confirmation captures breakout moves in trending markets while avoiding false
# breakouts in ranging conditions. Works in both bull and bear markets by
# trading breakouts in the direction of the prevailing 1-week trend.
# Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_price_channel_breakout_volume_v1"
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
    open_prices = prices['open'].values
    
    # Get daily data (same as primary for Donchian calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper channel = highest high of last 20 days
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel = lowest low of last 20 days
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (no shift needed as we use same timeframe)
    upper_channel_aligned = upper_channel  # Already daily
    lower_channel_aligned = lower_channel  # Already daily
    
    # Calculate 1-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower channel or trend turns bearish
            if close[i] < lower_channel_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above upper channel or trend turns bullish
            if close[i] > upper_channel_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume and bullish trend
            if (close[i] > upper_channel_aligned[i] and 
                open_prices[i] <= upper_channel_aligned[i] and  # Ensure breakout happened this bar
                vol_confirm[i] and
                close[i] > ema_1w_aligned[i]):  # Only long in bullish 1-week trend
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume and bearish trend
            elif (close[i] < lower_channel_aligned[i] and 
                  open_prices[i] >= lower_channel_aligned[i] and  # Ensure breakdown happened this bar
                  vol_confirm[i] and
                  close[i] < ema_1w_aligned[i]):  # Only short in bearish 1-week trend
                position = -1
                signals[i] = -0.25
    
    return signals