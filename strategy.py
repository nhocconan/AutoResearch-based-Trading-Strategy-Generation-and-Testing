#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
    # Trade only in direction of 1w EMA50 to avoid counter-trend whipsaws
    # Volume spike (>1.5x 20-period average) confirms participation
    # Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
    # Works in bull/bear markets by only trading with the dominant 1w trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian channels (20-period) on 1w
    # Upper = max(high_1w over last 20 periods)
    # Lower = min(low_1w over last 20 periods)
    donchian_upper = np.full(len(df_1w), np.nan)
    donchian_lower = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        donchian_upper[i] = np.max(high_1w[i-20:i])
        donchian_lower[i] = np.min(low_1w[i-20:i])
    
    # Get 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w volume for confirmation (>1.5x 20-period average)
    vol_ma_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-20:i])
    volume_spike_1w = volume_1w > (1.5 * vol_ma_1w)
    
    # Align all indicators to LTF (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # 1w trend filter
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: price returns to opposite Donchian level (mean reversion)
        long_exit = close[i] < donchian_lower_aligned[i] or not bullish_trend
        short_exit = close[i] > donchian_upper_aligned[i] or not bearish_trend
        
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

name = "1d_1w_donchian_breakout_ema50_volume_v1"
timeframe = "1d"
leverage = 1.0