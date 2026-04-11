#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# - Donchian levels from 6h: upper/lower bands act as dynamic support/resistance
# - Weekly pivot direction from 1d: long when price > weekly pivot (bullish bias), short when price < weekly pivot (bearish bias)
# - Volume confirmation: current 6h volume > 1.5x 20-period average of 1d volume (aligned)
# - Works in both bull (breakouts with volume in bullish bias) and bear (breakdowns with bearish bias) markets
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn

name = "6h_1d_donchian_weekly_pivot_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for weekly pivot and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d weekly pivot (using prior week's high, low, close)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using rolling window of 5 days (1 week)
    # We need prior week's values, so shift by 5
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5)
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5)
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_values = weekly_pivot.values
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_values)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_close > donchian_upper[i-1]  # Close above previous period's upper band
        breakout_short = price_close < donchian_lower[i-1]  # Close below previous period's lower band
        
        # Weekly pivot bias: long when price > weekly pivot, short when price < weekly pivot
        pivot_bias_long = price_close > weekly_pivot_aligned[i]
        pivot_bias_short = price_close < weekly_pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian upper breakout + bullish pivot bias + volume confirmation
        if breakout_long and pivot_bias_long and vol_confirm:
            enter_long = True
        
        # Short: Donchian lower breakdown + bearish pivot bias + volume confirmation
        if breakout_short and pivot_bias_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or pivot bias flip
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below lower band OR pivot bias flips to short
            exit_long = (price_close < donchian_lower[i-1]) or (not pivot_bias_long)
        elif position == -1:
            # Exit short if price breaks above upper band OR pivot bias flips to long
            exit_short = (price_close > donchian_upper[i-1]) or (not pivot_bias_short)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals