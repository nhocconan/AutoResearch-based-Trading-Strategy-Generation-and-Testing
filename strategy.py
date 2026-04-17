#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
# Enters long when price breaks above Donchian upper band with ADX > 25 (trending) and volume spike.
# Enters short when price breaks below Donchian lower band with ADX > 25 and volume spike.
# Exits when price returns to Donchian middle (mean) or ADX < 20 (trend weakening).
# Designed to capture strong trends with low turnover (target: 20-50 trades/year).
# Works in bull markets (breakout momentum) and bear markets (strong downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h
    # True Range = max(high-low, abs(high-previous close), abs(low-previous close))
    # Plus Directional Movement = high - previous high (if > previous low - low and > 0)
    # Minus Directional Movement = previous low - low (if > high - previous high and > 0)
    # ADX = smoothed average of DX
    
    # Calculate True Range
    high_low = high_12h - low_12h
    high_close = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    low_close = np.abs(np.diff(low_12h, prepend=low_12h[0]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate Directional Movement
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    # Fix first values
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align 4h indicators to main timeframe (4h)
    donchian_upper_4h = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_4h = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_4h = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Align 12h ADX to main timeframe (4h)
    adx_4h = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_4h[i]) or 
            np.isnan(donchian_lower_4h[i]) or 
            np.isnan(donchian_middle_4h[i]) or 
            np.isnan(adx_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: ADX > 25 (strong trend)
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20  # Exit when trend weakens
        
        # Price relative to Donchian channels
        price_above_upper = close[i] > donchian_upper_4h[i]
        price_below_lower = close[i] < donchian_lower_4h[i]
        price_above_middle = close[i] > donchian_middle_4h[i]
        price_below_middle = close[i] < donchian_middle_4h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume and strong trend
            if (price_above_upper and volume_filter and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume and strong trend
            elif (price_below_lower and volume_filter and strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to middle OR trend weakens
            if (price_below_middle or weak_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to middle OR trend weakens
            if (price_above_middle or weak_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hADX25_Volume"
timeframe = "4h"
leverage = 1.0