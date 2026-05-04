#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation
# Uses 4h Donchian(20) for breakout signals, 1d EMA50 for trend direction, and 1.5x volume spike for confirmation
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag on 1h timeframe
# Works in both bull and bear markets by following the 1d trend direction and using Donchian for structure
# Prioritizes BTC/ETH performance

name = "1h_Donchian20_1dEMA50_Trend_Volume"
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
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_roll_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (use previous completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, high_roll_max)  # Upper band
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, low_roll_min)   # Lower band
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Donchian breakout with 1d trend filter
        # Long: Price breaks above 4h Donchian upper band + volume spike + price above 1d EMA50 (uptrend)
        # Short: Price breaks below 4h Donchian lower band + volume spike + price below 1d EMA50 (downtrend)
        if position == 0:
            if (close[i] > donchian_high_aligned[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            elif (close[i] < donchian_low_aligned[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below 4h Donchian lower band OR price below 1d EMA50 (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above 4h Donchian upper band OR price above 1d EMA50 (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals