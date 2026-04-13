#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA200 trend filter
    # Long: price breaks above Donchian upper (20) AND volume > 1.5x 20-period average AND price > 1d EMA200
    # Short: price breaks below Donchian lower (20) AND volume > 1.5x 20-period average AND price < 1d EMA200
    # Exit: opposite Donchian breakout (lower for long exit, upper for short exit)
    # Using 1d EMA200 for strong trend filter to avoid counter-trend trades in choppy/bear markets
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 20-50 trades/year (~80-200 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        upper[i] = np.max(high[i-donchian_period:i])
        lower[i] = np.min(low[i-donchian_period:i])
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1d EMA200, only short if price < 1d EMA200
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > upper[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < lower[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: opposite Donchian breakout
        long_exit = close[i] < lower[i]
        short_exit = close[i] > upper[i]
        
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

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0