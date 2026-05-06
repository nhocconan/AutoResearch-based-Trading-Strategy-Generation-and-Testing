#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above upper Donchian band AND 1d close > 1d EMA50 (uptrend) AND volume > 1.8 * 20-bar avg volume
# Short when price breaks below lower Donchian band AND 1d close < 1d EMA50 (downtrend) AND volume > 1.8 * 20-bar avg volume
# Exit when price retraces to the midpoint of the Donchian channel (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe
# 1d EMA50 provides strong trend filter between 6h and 1d for better regime adaptation
# Volume threshold increased to 1.8x to reduce false breakouts and lower trade frequency
# Donchian midpoint exit works in ranging markets and captures mean reversion after breakout failure
# This strategy focuses on BTC and ETH as primary targets, avoiding SOL-only bias

name = "6h_Donchian20_1dEMA50_VolumeSpike_v1"
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
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    # Middle band = (upper + lower) / 2
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_band = (upper_band + lower_band) / 2.0
    
    # Shift by 1 to use only completed bar data (no look-ahead)
    upper_band_prev = np.roll(upper_band, 1)
    lower_band_prev = np.roll(lower_band, 1)
    middle_band_prev = np.roll(middle_band, 1)
    upper_band_prev[0] = np.nan
    lower_band_prev[0] = np.nan
    middle_band_prev[0] = np.nan
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_band_prev[i]) or np.isnan(lower_band_prev[i]) or 
            np.isnan(middle_band_prev[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper band AND uptrend AND volume spike
            if close[i] > upper_band_prev[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND downtrend AND volume spike
            elif close[i] < lower_band_prev[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to middle band (mean reversion)
            if close[i] <= middle_band_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to middle band (mean reversion)
            if close[i] >= middle_band_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals