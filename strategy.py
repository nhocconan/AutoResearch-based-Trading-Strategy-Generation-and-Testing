#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
    # Long: price breaks above 20-period high + price > 12h EMA50 + volume > 1.5x 20-bar avg
    # Short: price breaks below 20-period low + price < 12h EMA50 + volume > 1.5x 20-bar avg
    # Exit: opposite Donchian breakout or trend reversal
    # Works in bull/bear markets by only trading with 12h trend, reducing whipsaws
    # Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume MA for confirmation
    vol_ma_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    volume_spike_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Align 12h indicators to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # 12h trend filter
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: opposite breakout or trend reversal
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