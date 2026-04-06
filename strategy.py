#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with Donchian(20) breakout and volume confirmation
# In trending markets (CHOP < 38.2): trade breakouts in trend direction
# In ranging markets (CHOP > 61.8): fade extremes at Donchian bands
# Uses 1w trend filter to avoid counter-trend trades in strong trends
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_chop_donchian_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Choppiness Index (14-period) for regime detection
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    chop = 100 * np.log10(atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop_values = chop.values
    
    # 1-week EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(chop_values[i]) or np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        chop_val = chop_values[i]
        vol_ok = volume[i] > volume_threshold[i]
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Regime-based entries
            if vol_ok:
                # Trending regime (CHOP < 38.2): trade breakouts with trend filter
                if chop_val < 38.2:
                    # Long breakout in uptrend
                    if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                        close[i] > weekly_ema_aligned[i]):
                        signals[i] = 0.25
                        position = 1
                    # Short breakdown in downtrend
                    elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                          close[i] < weekly_ema_aligned[i]):
                        signals[i] = -0.25
                        position = -1
                # Ranging regime (CHOP > 61.8): fade extremes
                elif chop_val > 61.8:
                    # Short at upper band in range
                    if close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]:
                        signals[i] = -0.25
                        position = -1
                    # Long at lower band in range
                    elif close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]:
                        signals[i] = 0.25
                        position = 1
    
    return signals