#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 12h close > 12h EMA50 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below Donchian lower band AND 12h close < 12h EMA50 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price retraces to the Donchian midpoint (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 12h EMA50 provides strong trend filter for better regime adaptation in both bull and bear markets
# Volume threshold increased to 2.0x to significantly reduce false breakouts and lower trade frequency
# Donchian midpoint exit works in ranging markets and captures mean reversion after breakout failure

name = "6h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels for 6h timeframe (based on previous 20 bars)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Midpoint = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    midpoint = (upper_band + lower_band) / 2.0
    
    # Shift by 1 to use only completed bar data (no look-ahead)
    upper_band_prev = np.roll(upper_band, 1)
    lower_band_prev = np.roll(lower_band, 1)
    midpoint_prev = np.roll(midpoint, 1)
    upper_band_prev[0] = np.nan
    lower_band_prev[0] = np.nan
    midpoint_prev[0] = np.nan
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_band_prev[i]) or np.isnan(lower_band_prev[i]) or 
            np.isnan(midpoint_prev[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper band AND uptrend AND volume spike
            if close[i] > upper_band_prev[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND downtrend AND volume spike
            elif close[i] < lower_band_prev[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to midpoint (mean reversion)
            if close[i] <= midpoint_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to midpoint (mean reversion)
            if close[i] >= midpoint_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals