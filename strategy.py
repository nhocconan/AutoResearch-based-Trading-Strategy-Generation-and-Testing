#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with volume confirmation and weekly EMA trend filter
# Uses Donchian(20) breakout for trend capture, volume filter to ensure institutional participation,
# and weekly EMA to filter counter-trend trades. Works in both bull and bear markets by only
# taking trades in the direction of the higher timeframe trend. Low frequency design targets
# 20-50 trades per year to minimize fee drag.

name = "4h_donchian20_1w_ema_volume_v2"
timeframe = "4h"
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
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower (trend reversal) or hits upper (take profit)
            if close[i] < donchian_lower[i] or close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper (trend reversal) or hits lower (take profit)
            if close[i] > donchian_upper[i] or close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long: break above Donchian upper with volume confirmation and uptrend
            if close[i] > donchian_upper[i] and vol_confirm and uptrend:
                position = 1
                signals[i] = 0.25
            # Short: break below Donchian lower with volume confirmation and downtrend
            elif close[i] < donchian_lower[i] and vol_confirm and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals