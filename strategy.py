#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
    # Trade only in direction of 12h EMA50 to avoid counter-trend whipsaws
    # Volume spike (>1.5x 20-period average) confirms participation
    # Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
    # Works in bull/bear markets by only trading with the dominant 12h trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate previous 12h bar's Donchian channels (20-period)
    # Upper = max(high_prev_20), Lower = min(low_prev_20)
    high_roll_max = np.full(len(df_12h), np.nan)
    low_roll_min = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        high_roll_max[i] = np.max(high_12h[i-20:i])
        low_roll_min[i] = np.min(low_12h[i-20:i])
    
    # Get 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    volume_spike_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Align all indicators to LTF (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, high_roll_max)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, low_roll_min)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # 12h trend filter
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: opposite Donchian breakout or trend reversal
        long_exit = short_breakout or not bullish_trend
        short_exit = long_breakout or not bearish_trend
        
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