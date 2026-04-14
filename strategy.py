#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily range expansion breakouts with volume confirmation
# - Uses previous day's high-low range to set adaptive breakout levels (adapts to volatility)
# - Requires volume > 1.5x 24-period average for institutional confirmation
# - Filters for high volatility regimes using 80th percentile of daily range/price ratio
# - Designed to capture volatility expansion in both bull and bear markets
# - Discrete position sizing (0.25) to minimize churn and manage drawdown
# - Target: 100-150 trades over 4 years (25-38/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily range (high - low) for volatility measurement
    daily_range = high_1d - low_1d
    daily_range_series = pd.Series(daily_range)
    
    # 4h volume filter: current volume > 1.5x 24-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    # 4h Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate daily volatility as range normalized by price
    daily_volatility = daily_range / close_1d
    daily_vol_series = pd.Series(daily_volatility)
    # Use 80th percentile of daily volatility over 10 days as threshold (more selective)
    vol_threshold = daily_vol_series.rolling(window=10, min_periods=10).quantile(0.80).values
    # Align volatility threshold to 4h timeframe
    vol_threshold_4h = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(vol_threshold_4h[i]):
            continue
        
        # Get previous day's data for range-based breakout levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_range = prev_high - prev_low
        
        # Calculate breakout levels: previous day's high/low ± 0.2 * range
        upper_break = prev_high + 0.2 * prev_range
        lower_break = prev_low - 0.2 * prev_range
        
        # Create arrays for alignment
        upper_array = np.full(len(df_1d), upper_break)
        lower_array = np.full(len(df_1d), lower_break)
        
        upper_4h = align_htf_to_ltf(prices, df_1d, upper_array)[i]
        lower_4h = align_htf_to_ltf(prices, df_1d, lower_array)[i]
        
        if position == 0:
            # Long: Price breaks above upper level with volume and high volatility regime
            if (close[i] > upper_4h and close[i-1] <= upper_4h and 
                volume[i] > vol_ma[i] * 1.5 and 
                daily_volatility[i] > vol_threshold_4h[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower level with volume and high volatility regime
            elif (close[i] < lower_4h and close[i-1] >= lower_4h and 
                  volume[i] > vol_ma[i] * 1.5 and 
                  daily_volatility[i] > vol_threshold_4h[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below lower level (reversal) or drops below Donchian low
            if close[i] < lower_4h or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above upper level (reversal) or rises above Donchian high
            if close[i] > upper_4h or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_RangeExpansion_Breakout_Volume_v2"
timeframe = "4h"
leverage = 1.0