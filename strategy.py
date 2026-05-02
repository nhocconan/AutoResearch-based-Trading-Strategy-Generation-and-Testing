#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel (20-period high/low) from 4h OHLC for institutional breakout zones
# 1d EMA50 ensures alignment with medium-term trend to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Designed for 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes)
# BTC and ETH focused with SOL as secondary validation

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 20-period average (20*4h = 80h ~ 3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate 4h Donchian levels (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    
    # Donchian: Upper = 20-period high, Lower = 20-period low
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (wait for 4h close)
    donchian_upper_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA50 and Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above upper DONCHIAN AND price > 1d EMA50 (bullish trend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower DONCHIAN AND price < 1d EMA50 (bearish trend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower DONCHIAN (reversion to mean) OR price below 1d EMA50 (trend change)
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper DONCHIAN (reversion to mean) OR price above 1d EMA50 (trend change)
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals