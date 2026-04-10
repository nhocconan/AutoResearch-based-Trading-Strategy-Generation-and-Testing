#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Williams Fractals from 1d: bearish fractal (sell signal) when middle high is highest of 5 bars
# - Bullish fractal (buy signal) when middle low is lowest of 5 bars
# - 1d ADX(14) > 25 to ensure trending market and avoid chop
# - Volume confirmation: current 12h volume > 2.0x 20-period average
# - ATR-based trailing stop: exit long when price < highest_high - 3.0*ATR, exit short when price > lowest_low + 3.0*ATR
# - Designed for 12h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_williams_fractal_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: need 5 bars (2 left, 2 right)
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: middle high is highest of 5
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: middle low is lowest of 5
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Pre-compute 1d ADX(14) for trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(high_1d, 1))
    tr3 = np.abs(low_1d - np.roll(low_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe with extra delay for fractals (need 2 extra bars for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 12h Donchian channels (20-period) for exit
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian upper band: highest high over past 20 periods
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low over past 20 periods
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (2.0 * avg_volume_20)
    
    # Pre-compute 12h ATR(14) for trailing stop
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_12h[i] > highest_high:
                highest_high = close_12h[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_12h[i] < highest_high - 3.0 * atr_14[i] or close_12h[i] < highest_20[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_12h[i] < lowest_low:
                lowest_low = close_12h[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_12h[i] > lowest_low + 3.0 * atr_14[i] or close_12h[i] > lowest_20[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams Fractal breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: price closes above bullish fractal level
                if not np.isnan(bullish_fractal_aligned[i]) and close_12h[i] > bullish_fractal_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    highest_high = close_12h[i]
                    signals[i] = 0.25
                # Breakout short: price closes below bearish fractal level
                elif not np.isnan(bearish_fractal_aligned[i]) and close_12h[i] < bearish_fractal_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    lowest_low = close_12h[i]
                    signals[i] = -0.25
    
    return signals