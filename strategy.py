#!/usr/bin/env python3
# 12h_WilliamsFractal_Reversal
# Hypothesis: Williams Fractal reversals on daily chart with volume spike confirmation and ADX trend filter on 12h timeframe.
# In bear markets, sell at bearish fractals (resistance) with volume; in bull markets, buy at bullish fractals (support) with volume.
# ADX > 25 ensures we only trade in trending conditions, avoiding whipsaws in ranging markets.
# Uses 1-day Williams Fractals (requires 2-bar confirmation) aligned to 12h.
# Target: 20-40 trades/year to minimize fee drag. Works in both bull and bear by fading false breakouts at fractal levels.

name = "12h_WilliamsFractal_Reversal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected import name

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # --- Williams Fractals on Daily (requires 2-bar confirmation) ---
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n-2] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n-2] and low[n] < low[n+2]
    # We need 2 bars after the center for confirmation, so additional_delay_bars=2
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True

    # --- ADX Trend Filter on Daily ---
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
        def WilderSmoothing(data, period):
            result = np.zeros_like(data)
            result[period-1] = np.mean(data[0:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = WilderSmoothing(tr, period)
        dm_plus_smooth = WilderSmoothing(dm_plus, period)
        dm_minus_smooth = WilderSmoothing(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = WilderSmoothing(dx, period)
        return adx

    adx_1d = calculate_adx(high_1d, low_1d, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # --- Volume Confirmation on 12h ---
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)  # At least 2x average volume

    # --- Align HTF indicators to 12h ---
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )

    # --- Generate Signals ---
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade when ADX > 25 (trending market)
        if adx_1d_aligned[i] > 25:
            if position == 0:
                # SHORT at bearish fractal (resistance) with volume spike
                if bearish_fractal_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                # LONG at bullish fractal (support) with volume spike
                elif bullish_fractal_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: price closes below bullish fractal level or ADX weakens
                if adx_1d_aligned[i] < 20:  # Trend weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: price closes above bearish fractal level or ADX weakens
                if adx_1d_aligned[i] < 20:  # Trend weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging market (ADX <= 25), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals