#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period average)
    # Uses 12h EMA50 for primary trend direction to avoid counter-trend whipsaws in bear markets
    # Volume spike confirms institutional participation on breakouts
    # Exits on return to Donchian midpoint for mean reversion
    # Tight entry conditions target 20-30 trades/year (80-120 total over 4 years) to minimize fee drag
    # Works in bull markets via trend-following breakouts, in bear via faded counter-trend spikes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate previous 4h bar's Donchian levels (20-period)
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    lookback = 20
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(low_4h), np.nan)
    
    for i in range(lookback, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-lookback:i])
        lower_4h[i] = np.min(low_4h[i-lookback:i])
    
    # Get 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_4h[i] = np.mean(volume[i-20:i])
    volume_spike_4h = volume > (1.5 * vol_ma_4h)
    
    # Align all indicators to LTF (4h)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_4h_aligned[i]
        short_breakout = close[i] < lower_4h_aligned[i]
        
        # 12h trend filter
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_4h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_4h[i]
        
        # Exit logic: price returns to Donchian midpoint
        donchian_mid = (upper_4h + lower_4h) / 2
        donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
        
        # Exit when price returns to midpoint (within 0.2% tolerance)
        midpoint_distance = abs(close[i] - donchian_mid_aligned[i]) / close[i]
        at_midpoint = midpoint_distance < 0.002
        
        long_exit = at_midpoint or not bullish_trend
        short_exit = at_midpoint or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0