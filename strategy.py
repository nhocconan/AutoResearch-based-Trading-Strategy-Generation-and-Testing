#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + Volume Spike + ADX Trend Filter
Hypothesis: Williams Fractals identify key support/resistance levels. Breakouts above/below these levels with volume confirmation and ADX > 25 capture strong trending moves. Works in both bull (breakouts to new highs) and bear (breakdowns to new lows) markets. Low trade frequency due to strict fractal confirmation and volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d (need 2 extra bars for confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Apply additional delay for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate ADX on 1d for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values with Wilder smoothing
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values use Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    plus_di_1d = 100 * wilders_smooth(plus_dm, 14) / np.where(atr_1d != 0, atr_1d, 1)
    minus_di_1d = 100 * wilders_smooth(minus_dm, 14) / np.where(atr_1d != 0, atr_1d, 1)
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    adx_1d = wilders_smooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike: current volume > 2.0x 20-period average (using 1d volume)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.zeros_like(vol_1d)
    for i in range(len(vol_1d)):
        if i < 20:
            vol_ma_1d[i] = np.mean(vol_1d[max(0, i-19):i+1]) if i >= 0 else vol_1d[i]
        else:
            vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_spike_1d = vol_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        bullish_fractal_level = bullish_fractal_aligned[i]
        bearish_fractal_level = bearish_fractal_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_ok = vol_spike_1d_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above bullish fractal (resistance) + ADX > 25 + volume spike
            if (close[i] > bullish_fractal_level and 
                adx_val > 25 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below bearish fractal (support) + ADX > 25 + volume spike
            elif (close[i] < bearish_fractal_level and 
                  adx_val > 25 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below bullish fractal or ADX weakens
            if close[i] < bullish_fractal_level or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above bearish fractal or ADX weakens
            if close[i] > bearish_fractal_level or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0