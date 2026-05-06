#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly (1w) price channel breakout with daily trend confirmation
# - Uses 1w Donchian channel (20-period) to identify volatility breakouts
# - Uses 1d EMA(50) to confirm trend direction and avoid counter-trend entries
# - Requires volume spike (>2x 20-period average) for confirmation
# - Enters long when price breaks above 1w high with volume spike and price > 1d EMA50
# - Enters short when price breaks below 1w low with volume spike and price < 1d EMA50
# - Exits when price crosses back below/above 1w close or opposite breakout occurs
# - Designed to capture strong trending moves after weekly consolidation with daily trend filter
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_1wDonchian_1dEMA50_Volume_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w Donchian Channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filters (1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    # Align 1w indicators to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, donchian_low)
    close_1w_1d = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Align 1d EMA50 to 1d timeframe (already aligned, but for consistency)
    ema_50_1d = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(close_1w_1d[i]) or np.isnan(ema_50_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w high with volume spike and price > 1d EMA50
            if close[i] > donchian_high_1d[i] and volume_spike[i] and close[i] > ema_50_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w low with volume spike and price < 1d EMA50
            elif close[i] < donchian_low_1d[i] and volume_spike[i] and close[i] < ema_50_1d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w close OR short signal occurs
            if close[i] < close_1w_1d[i] or (close[i] < donchian_low_1d[i] and volume_spike[i] and close[i] < ema_50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w close OR long signal occurs
            if close[i] > close_1w_1d[i] or (close[i] > donchian_high_1d[i] and volume_spike[i] and close[i] > ema_50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals