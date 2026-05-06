#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d trend filter
# - Uses 4h Donchian channel (20-period) to identify breakout levels
# - Uses 1d EMA (50) to determine long-term trend direction (only long in uptrend, short in downtrend)
# - Requires volume spike (>2x 20-period average) for breakout confirmation
# - Exits when price returns to the 4h Donchian midpoint or on opposite breakout
# - Designed to capture strong trending moves with proper filtering for both bull and bear markets
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing

name = "1h_DonchianBreakout_4hChannel_1dTrend_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian Channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band (20-period high)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band (20-period low)
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle band (midpoint)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d EMA (50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_1h = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Align 1d EMA to 1h timeframe
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filters (1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(donchian_mid_1h[i]) or np.isnan(ema_50_1h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high + volume spike + uptrend (price > 1d EMA50)
            if close[i] > donchian_high_1h[i] and volume_spike[i] and close[i] > ema_50_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low + volume spike + downtrend (price < 1d EMA50)
            elif close[i] < donchian_low_1h[i] and volume_spike[i] and close[i] < ema_50_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h Donchian midpoint OR price breaks below Donchian low (contrarian signal)
            if close[i] <= donchian_mid_1h[i] or close[i] < donchian_low_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Donchian midpoint OR price breaks above Donchian high (contrarian signal)
            if close[i] >= donchian_mid_1h[i] or close[i] > donchian_high_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals