#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and chop regime filter
# Long when price breaks above 1w Donchian upper (20-period) AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range)
# Short when price breaks below 1w Donchian lower (20-period) AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range)
# Exit when price crosses 1w EMA50 (trend reversal) OR chop < 38.2 (trending regime)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe
# 1w Donchian provides clear structure with proven breakout edge
# Volume confirmation filters weak breakouts
# Chop regime filter avoids whipsaws in strong trends and captures mean reversion in ranges
# Works in bull (breakouts above upper in uptrend) and bear (breakdowns below lower in downtrend)

name = "1d_1wDonchian20_VolumeChop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for Donchian calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period) based on previous 1w bar
    donchian_upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for exit signal
    close_series_1w = pd.Series(close_1w)
    ema_50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w Donchian levels and EMA to 1d timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate Chopiness Index (14-period) on 1d timeframe for regime filter
    def calculate_chop(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1))))
        atr.iloc[0] = high[0] - low[0]  # First ATR
        atr_sum = atr.rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop.values
    
    chop = calculate_chop(high, low, close, 14)
    chop_range = chop > 61.8  # Range regime (mean revert)
    chop_trend = chop < 38.2  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper with volume confirmation AND in range regime
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and chop_range[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with volume confirmation AND in range regime
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and chop_range[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 OR chop indicates trending regime
            if close[i] < ema_50_aligned[i] or chop_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 OR chop indicates trending regime
            if close[i] > ema_50_aligned[i] or chop_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals