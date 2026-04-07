#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian(20) breakout with weekly trend filter and daily volume confirmation
# Hypothesis: Breakouts in direction of weekly trend capture strong trends; volume confirmation avoids false breakouts.
# Works in bull via breakouts with trend, in bear via avoiding counter-trend breakouts.
# Target: 10-25 trades/year to minimize fee drift.
name = "1d_donchian20_1w_volume_trend_v1"
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
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian channels (20-period high/low) on daily
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches opposite band OR trend reverses
            if close[i] <= lowest_low[i] or not price_above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches opposite band OR trend reverses
            if close[i] >= highest_high[i] or not price_below_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + volume confirmation + above weekly EMA
            if close[i] > highest_high[i] and vol_confirm and price_above_weekly_ema:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + volume confirmation + below weekly EMA
            elif close[i] < lowest_low[i] and vol_confirm and price_below_weekly_ema:
                position = -1
                signals[i] = -0.25
    
    return signals