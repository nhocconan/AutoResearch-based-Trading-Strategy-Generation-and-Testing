#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams Fractals with volume and ADX filter
# Long when price breaks above bearish fractal with volume > 1.5x average and ADX > 25
# Short when price breaks below bullish fractal with volume > 1.5x average and ADX > 25
# Williams Fractals identify key support/resistance levels. Volume confirms breakout strength.
# ADX filter ensures trades occur in trending markets, reducing false breakouts in ranging conditions.
# Works in bull/bear markets: breakouts capture momentum, ADX filter avoids ranging whipsaws.
# Target: 20-50 trades per year (80-200 over 4 years) with 0.25 position sizing.

name = "4h_1dWilliamsFractal_Volume_ADX_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Williams Fractals ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Williams Fractals: bearish (high) and bullish (low)
    high_vals = df_1d['high'].values
    low_vals = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_vals), np.nan)
    bullish_fractal = np.full(len(low_vals), np.nan)
    
    # Bearish fractal: middle bar has highest high, 2 bars on each side lower
    for i in range(2, len(high_vals) - 2):
        if (high_vals[i] > high_vals[i-1] and high_vals[i] > high_vals[i-2] and
            high_vals[i] > high_vals[i+1] and high_vals[i] > high_vals[i+2]):
            bearish_fractal[i] = high_vals[i]
    
    # Bullish fractal: middle bar has lowest low, 2 bars on each side higher
    for i in range(2, len(low_vals) - 2):
        if (low_vals[i] < low_vals[i-1] and low_vals[i] < low_vals[i-2] and
            low_vals[i] < low_vals[i+1] and low_vals[i] < low_vals[i+2]):
            bullish_fractal[i] = low_vals[i]
    
    # Align fractals to 4h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # ADX calculation on 1-day timeframe
    # Calculate True Range
    tr1 = high_vals[1:] - low_vals[1:]
    tr2 = np.abs(high_vals[1:] - np.append([np.nan], high_vals[:-1])[1:])
    tr3 = np.abs(low_vals[1:] - np.append([np.nan], low_vals[:-1])[1:])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, np.nan)  # align with original index
    
    # Calculate Directional Movement
    dm_plus = np.where((high_vals[1:] - high_vals[:-1]) > (low_vals[:-1] - low_vals[1:]), 
                       np.maximum(high_vals[1:] - high_vals[:-1], 0), 0)
    dm_minus = np.where((low_vals[:-1] - low_vals[1:]) > (high_vals[1:] - high_vals[:-1]), 
                        np.maximum(low_vals[:-1] - low_vals[1:], 0), 0)
    dm_plus = np.insert(dm_plus, 0, np.nan)
    dm_minus = np.insert(dm_minus, 0, np.nan)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA-like)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above bearish fractal with volume and ADX confirmation
            if close[i] > bearish_fractal_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below bullish fractal with volume and ADX confirmation
            elif close[i] < bullish_fractal_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish fractal or ADX weakens
            if close[i] < bullish_fractal_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish fractal or ADX weakens
            if close[i] > bearish_fractal_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals