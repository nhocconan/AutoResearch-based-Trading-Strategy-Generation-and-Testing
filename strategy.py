#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter
# Donchian channels provide objective breakout levels with proven edge in crypto
# Volume spike confirms breakout strength, reducing false signals
# 12h EMA50 ensures alignment with medium-term trend, avoiding counter-trend trades
# Designed for low frequency: ~25-40 trades/year per symbol with discrete sizing
# Works in bull/bear: EMA filter adapts to trend direction, volume confirms institutional interest

name = "4h_Donchian20_Volume_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from 4h data
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA (balanced for trade frequency)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 50)  # Need Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian upper with volume spike and price > 12h EMA50
            if close[i] > donchian_upper[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower with volume spike and price < 12h EMA50
            elif close[i] < donchian_lower[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to Donchian lower or opposite breakout with volume
            if close[i] < donchian_lower[i] or (close[i] < donchian_lower[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Donchian upper or opposite breakout with volume
            if close[i] > donchian_upper[i] or (close[i] > donchian_upper[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals