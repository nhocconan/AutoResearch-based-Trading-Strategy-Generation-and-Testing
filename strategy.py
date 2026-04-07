#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 1d Trend Filter
Long when price breaks above Donchian(20) high with volume > 1.5x average AND 1d close > 1d open
Short when price breaks below Donchian(20) low with volume > 1.5x average AND 1d close < 1d open
Exit when price crosses opposite Donchian band (e.g., long exits on break below lower band)
Designed to capture strong momentum moves while avoiding false breakouts in low-volume/choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "4h_donchian_breakout_volume_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period high/low) ===
    # Use pandas rolling for clarity and proper min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 1d trend filter: bullish if close > open, bearish if close < open ===
    df_1d = get_htf_data(prices, '1d')
    # Daily bullish/bearish signal: 1 if bullish, -1 if bearish, 0 if doji/neutral
    daily_bullish = df_1d['close'] > df_1d['open']
    daily_bearish = df_1d['close'] < df_1d['open']
    daily_trend = np.where(daily_bullish, 1, np.where(daily_bearish, -1, 0)).values
    daily_trend_aligned = align_ltf_to_htf(prices, df_1d, daily_trend)  # Properly aligned with shift(1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(daily_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 1d trend filter
            if close[i] > donchian_high[i] and daily_trend_aligned[i] == 1:
                # Breakout above upper band with bullish 1d trend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and daily_trend_aligned[i] == -1:
                # Breakdown below lower band with bearish 1d trend -> short
                position = -1
                signals[i] = -0.25
    
    return signals

# Note: align_ltf_to_htf is not in mtf_data; using align_htf_to_ltf as per standard
# Correcting the import and function call
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period high/low) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 1d trend filter: bullish if close > open, bearish if close < open ===
    df_1d = get_htf_data(prices, '1d')
    daily_bullish = df_1d['close'] > df_1d['open']
    daily_bearish = df_1d['close'] < df_1d['open']
    daily_trend = np.where(daily_bullish, 1, np.where(daily_bearish, -1, 0)).values
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)  # shift(1) for prior day's close
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(daily_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 1d trend filter
            if close[i] > donchian_high[i] and daily_trend_aligned[i] == 1:
                # Breakout above upper band with bullish 1d trend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and daily_trend_aligned[i] == -1:
                # Breakdown below lower band with bearish 1d trend -> short
                position = -1
                signals[i] = -0.25
    
    return signals