#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume spike filter
    # Long: price breaks above 4h Donchian high(20) AND price > 1d EMA50 AND volume > 1.5x 20-period average
    # Short: price breaks below 4h Donchian low(20) AND price < 1d EMA50 AND volume > 1.5x 20-period average
    # Exit: price crosses 4h EMA10 (mean reversion in 4h timeframe)
    # Using 1d for trend filter (HTF), 4h only for entry/exit timing
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 20-50 trades/year (~80-200 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and EMA10 (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high(20) = max(high_4h, lookback=20)
    # Donchian low(20) = min(low_4h, lookback=20)
    donch_high_4h = np.full(len(high_4h), np.nan)
    donch_low_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        donch_high_4h[i] = np.max(high_4h[i-20:i])
        donch_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian levels to 15m (wait for completed 4h bar)
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # 4h EMA10 for exit signal
    close_4h = df_4h['close'].values
    ema_10_4h = pd.Series(close_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_10_4h)
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or 
            np.isnan(ema_10_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1d EMA50, only short if price < 1d EMA50
        long_trend_ok = close[i] > ema_50_1d_aligned[i]
        short_trend_ok = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > donch_high_4h_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < donch_low_4h_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: price crosses 4h EMA10
        long_exit = close[i] < ema_10_4h_aligned[i]
        short_exit = close[i] > ema_10_4h_aligned[i]
        
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

name = "4h_1d_donchian_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0