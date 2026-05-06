#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 1w close > 1w EMA34 AND volume > 1.5 * 20-bar average volume
# Short when price breaks below Donchian(20) low AND 1w close < 1w EMA34 AND volume > 1.5 * 20-bar average volume
# Exit when price reverses to Donchian(20) midpoint OR trend filter fails
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian channels provide clear breakout levels with built-in stoploss at opposite channel
# 1w EMA34 provides strong multi-timeframe trend filter for better regime adaptation
# Volume confirmation reduces false signals during low participation periods
# Works in both bull and bear markets by following the primary 1w trend

name = "12h_Donchian20_1wEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels for 12h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: price breaks above Donchian high AND uptrend AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND downtrend AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend filter fails
            if close[i] < donchian_mid[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend filter fails
            if close[i] > donchian_mid[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals