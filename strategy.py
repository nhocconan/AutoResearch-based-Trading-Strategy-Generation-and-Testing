#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
    # Uses 1d EMA50 for medium-term trend direction, 6h Donchian for breakout signals
    # Volume spike (>2.0x 20-period average) confirms strong momentum
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    # Only trades with the dominant 1d trend to avoid counter-trend whipsaws
    # Donchian channels provide clear structure for breakouts in both bull and bear markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Get 6h volume for confirmation (>2.0x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (2.0 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # 1d EMA50 trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_6h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_6h[i]
        
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

name = "6h_1d_donchian_breakout_ema50_volume_v1"
timeframe = "6h"
leverage = 1.0