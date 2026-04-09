#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + 1d trend filter + volume confirmation
# 4h Donchian(20) captures medium-term breakouts; 1d EMA(50) defines trend direction
# Volume confirmation ensures breakout authenticity; session filter (08-20 UTC) reduces noise
# Discrete sizing 0.20 limits drawdown; target 60-150 trades over 4 years (15-37/year)
# Works in bull/bear: trend filter adapts, breakouts work in both directions

name = "1h_4h_1d_donchian_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = np.full(len(df_4h), np.nan)
    donchian_low_4h = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        if i < 20:
            donchian_high_4h[i] = np.nan
            donchian_low_4h[i] = np.nan
        else:
            donchian_high_4h[i] = np.max(high_4h[i-20:i])
            donchian_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 1h timeframe (wait for 4h bar close)
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 1h timeframe (wait for 1d bar close)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 20-period average volume for volume confirmation (1h timeframe)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if outside trading session or any required data is invalid
        if not in_session[i] or \
           (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < 4h Donchian low OR price < 1d EMA (trend change)
            if close[i] < donchian_low_4h_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > 4h Donchian high OR price > 1d EMA (trend change)
            if close[i] > donchian_high_4h_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + 1d EMA filter
            if volume_confirmed and in_session[i]:
                # Long entry: price > 4h Donchian high AND price > 1d EMA (bullish alignment)
                if close[i] > donchian_high_4h_aligned[i] and close[i] > ema_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price < 4h Donchian low AND price < 1d EMA (bearish alignment)
                elif close[i] < donchian_low_4h_aligned[i] and close[i] < ema_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals