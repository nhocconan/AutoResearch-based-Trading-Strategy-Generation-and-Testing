#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with weekly trend filter and volume confirmation
# Uses 1-day timeframe for signals, weekly trend for filtering to avoid counter-trend trades
# Designed to work in both bull and bear markets by only trading with the weekly trend
# Low trade frequency expected (10-25 trades/year) to minimize fee drag

name = "1d_1w_donchian_breakout_trend_filter_v1"
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
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly data for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1-day data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1-day data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-day high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-day low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1-day timeframe (wait for 1-day bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if Donchian levels or trend filter are not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals with volume confirmation
        if not vol_confirm[i]:
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1-day Donchian low (breakdown)
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 1-day Donchian high (breakout)
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade with weekly trend: long when price > weekly EMA50, short when price < weekly EMA50
            # Long entry: price breaks above 1-day Donchian high with volume and weekly uptrend
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 1-day Donchian low with volume and weekly downtrend
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals