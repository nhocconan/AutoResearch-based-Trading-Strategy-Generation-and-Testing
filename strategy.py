#!/usr/bin/env python3
# 6h_1d_donchian_weekly_pivot_volume
# Strategy: 6-hour Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Price breaks out of 6-hour Donchian channels in alignment with weekly pivot bias
# (above weekly pivot = long bias, below = short bias), confirmed by volume spikes.
# Works in bull markets via breakout continuation and in bear via mean reversion at pivot levels.
# Weekly pivot provides structural support/resistance that price respects across regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Weekly high/low/close from Friday's data (simplified: use rolling window)
    # For true weekly pivot, we need weekly OHLC - approximate using daily data
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We'll use 5-day rolling window for weekly approximation
    
    # Calculate rolling weekly high, low, close
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly support/resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot data to 6h timeframe (wait for daily close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Determine bias from weekly pivot
        bullish_bias = price_close > weekly_pivot_aligned[i]
        bearish_bias = price_close < weekly_pivot_aligned[i]
        
        # Breakout conditions with weekly pivot filter
        # Long: price breaks above Donchian high AND above weekly pivot (bullish bias)
        long_breakout = (price_high > donchian_high[i]) and bullish_bias
        # Short: price breaks below Donchian low AND below weekly pivot (bearish bias)
        short_breakout = (price_low < donchian_low[i]) and bearish_bias
        
        # Fade at extreme weekly levels (mean reversion)
        # Long: price touches/slightly breaks weekly S3 and shows rejection
        long_fade = (price_low <= weekly_s3_aligned[i] * 1.002) and bullish_bias
        # Short: price touches/slightly breaks weekly R3 and shows rejection
        short_fade = (price_high >= weekly_r3_aligned[i] * 0.998) and bearish_bias
        
        # Combine signals with volume confirmation
        long_signal = (long_breakout or long_fade) and vol_spike[i]
        short_signal = (short_breakout or short_fade) and vol_spike[i]
        
        # Exit conditions
        # Exit long when price returns to Donchian midpoint or breaks below weekly pivot
        exit_long = position == 1 and (price_close < donchian_mid[i] or price_close < weekly_pivot_aligned[i])
        # Exit short when price returns to Donchian midpoint or breaks above weekly pivot
        exit_short = position == -1 and (price_close > donchian_mid[i] or price_close > weekly_pivot_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals