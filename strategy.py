#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d ADX trend filter and volume confirmation
# Williams Fractals identify key reversal points; breakouts above bearish or below bullish fractals
# capture momentum with high probability. 1d ADX > 25 ensures trending markets to avoid whipsaws.
# Volume > 1.5x 20-period average confirms participation. Designed for 4h timeframe targeting
# 20-40 trades/year with robustness in both bull and bear markets via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Fractals and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (bearish: high surrounded by two lower highs on each side)
    # Bullish fractal: low surrounded by two higher lows on each side
    n1 = len(high_1d)
    bearish_fractal = np.full(n1, np.nan)
    bullish_fractal = np.full(n1, np.nan)
    
    for i in range(2, n1 - 2):
        # Bearish fractal: current high is highest of 5-bar window
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: current low is lowest of 5-bar window
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals require 2-bar confirmation after the pattern bar
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d ADX(14) for trend filter (ADX > 25 = trending)
    # Calculate +DI, -DI, DX
    plus_dm = np.zeros(n1)
    minus_dm = np.zeros(n1)
    tr = np.zeros(n1)
    
    for i in range(1, n1):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        tr[i] = max(high_1d[i] - low_1d[i],
                    abs(high_1d[i] - close_1d[i-1]),
                    abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / period
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above bearish fractal + ADX > 25 + volume confirmation
            if (close[i] > bearish_fractal_aligned[i] and
                adx_1d_aligned[i] > 25 and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal + ADX > 25 + volume confirmation
            elif (close[i] < bullish_fractal_aligned[i] and
                  adx_1d_aligned[i] > 25 and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite fractal level or ADX < 20 (trend weakening)
            if position == 1:
                # Exit long: price returns below bullish fractal or ADX < 20
                if (close[i] < bullish_fractal_aligned[i] or
                    adx_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns above bearish fractal or ADX < 20
                if (close[i] > bearish_fractal_aligned[i] or
                    adx_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_1dADX25_VolumeConfirm"
timeframe = "4h"
leverage = 1.0