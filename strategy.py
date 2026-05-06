#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d trend filter and volume confirmation
# - Uses 12h Donchian(20) for breakout signals
# - Uses 1d EMA50 for trend filter (long only above, short only below)
# - Uses 6h volume spike for entry confirmation
# - Designed to capture medium-term momentum with trend alignment
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing
# - Works in bull (breakouts with trend) and bear (breakouts against trend filtered out)

name = "6h_12hDonchian_20_1dEMA50_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (6h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (2.0 * vol_ma_10)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high AND above 1d EMA50 with volume spike
            if close[i] > donchian_high_6h[i] and close[i] > ema_50_1d_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low AND below 1d EMA50 with volume spike
            elif close[i] < donchian_low_6h[i] and close[i] < ema_50_1d_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h Donchian low OR breaks below 1d EMA50
            if close[i] < donchian_low_6h[i] or close[i] < ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h Donchian high OR breaks above 1d EMA50
            if close[i] > donchian_high_6h[i] or close[i] > ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals