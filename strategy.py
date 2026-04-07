#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Hypothesis: Daily breakouts with weekly trend alignment and volume confirmation capture strong trends.
# Works in bull via breakouts with trend, in bear via avoiding false breakouts in chop.
# Target: 10-25 trades/year to minimize fee drag.
name = "1d_donchian20_1w_trend_volume_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly 20-period EMA for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get daily volume for confirmation (use 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 20-day average volume
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches opposite band OR trend changes
            if close[i] <= lowest_low[i] or not price_above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches opposite band OR trend changes
            if close[i] >= highest_high[i] or not price_below_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + volume confirmation + uptrend
            if close[i] > highest_high[i] and vol_confirm and price_above_weekly_ema:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + volume confirmation + downtrend
            elif close[i] < lowest_low[i] and vol_confirm and price_below_weekly_ema:
                position = -1
                signals[i] = -0.25
    
    return signals