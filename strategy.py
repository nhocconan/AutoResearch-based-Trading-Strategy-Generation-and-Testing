#!/usr/bin/env python3

# Hypothesis: 12h timeframe with weekly price channel structure and daily momentum filter.
# Uses weekly Donchian channel (20-period) for breakout entries and 1d RSI for momentum confirmation.
# Weekly Donchian provides robust support/resistance that adapts to volatility, working in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
# Weekly data changes slowly, reducing whipsaw and improving win rate in ranging markets.

name = "12h_Donchian20_1dRSI_Momentum"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channel (20-period) from previous week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channel
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate 20-period Donchian bands
    upper_band = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (wait for weekly bar to close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Breakout conditions: price must close beyond the band
    breakout_up = close > upper_band_aligned
    breakout_down = close < lower_band_aligned
    
    # Get daily data for RSI momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI
    delta = np.diff(df_1d['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # Avoid division by zero
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Momentum filters: RSI > 50 for longs, RSI < 50 for shorts
    rsi_long_filter = rsi_aligned > 50
    rsi_short_filter = rsi_aligned < 50
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(rsi_long_filter[i]) or np.isnan(rsi_short_filter[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper band + RSI > 50 + volume filter
            if breakout_up[i] and rsi_long_filter[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band + RSI < 50 + volume filter
            elif breakout_down[i] and rsi_short_filter[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to lower band or RSI < 50
            if close[i] <= lower_band_aligned[i] or not rsi_long_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to upper band or RSI > 50
            if close[i] >= upper_band_aligned[i] or not rsi_short_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals