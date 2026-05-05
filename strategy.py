#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian channel breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian(20) upper band AND price > 12h EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian(20) lower band AND price < 12h EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price crosses 12h EMA50 in opposite direction OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian breakout provides clear structure-based entries in trending markets
# 12h EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "12h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channel (20-period)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe (wait for completed daily bar)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 1d Donchian upper band, above 12h EMA50, volume confirmation, in session
            if (close[i] > upper_band_aligned[i] and close[i-1] <= upper_band_aligned[i-1] and 
                close[i] > ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian lower band, below 12h EMA50, volume confirmation, in session
            elif (close[i] < lower_band_aligned[i] and close[i-1] >= lower_band_aligned[i-1] and 
                  close[i] < ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 12h EMA50 OR volume drops below average
            if close[i] < ema50_12h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 12h EMA50 OR volume drops below average
            if close[i] > ema50_12h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals