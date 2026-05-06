#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly price channels and volume confirmation
# - Uses 1w Donchian channels (20-period) for structural support/resistance
# - Uses 1d EMA (34) for intermediate trend direction
# - Enters long when price breaks above 1w upper channel with volume spike in bullish 1d trend
# - Enters short when price breaks below 1w lower channel with volume spike in bearish 1d trend
# - Exits when price returns to 1w channel midpoint or trend reverses
# - Designed to capture breakouts from weekly consolidation with daily trend alignment
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "6h_1wDonchian_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get 1d data for EMA and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d EMA (34)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w indicators to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_6h = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Align 1d EMA to 6h timeframe
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(donchian_mid_6h[i]) or np.isnan(ema_34_6h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout from weekly channel with volume and trend alignment
            bullish_trend = close[i] > ema_34_6h[i]
            bearish_trend = close[i] < ema_34_6h[i]
            
            # Long: break above weekly high with volume spike in bullish trend
            if bullish_trend and close[i] > donchian_high_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume spike in bearish trend
            elif bearish_trend and close[i] < donchian_low_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly midpoint OR trend turns bearish
            if close[i] < donchian_mid_6h[i] or close[i] < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly midpoint OR trend turns bullish
            if close[i] > donchian_mid_6h[i] or close[i] > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals