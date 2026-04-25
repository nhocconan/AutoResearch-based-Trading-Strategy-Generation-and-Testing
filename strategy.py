#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 12h ADX Trend + Volume Spike
Hypothesis: Williams fractals identify significant swing points on 12h timeframe.
Breakouts above bearish fractals or below bullish fractals with volume spike,
aligned with 12h ADX > 25 (strong trend), capture momentum in both bull and bear markets.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for fractals and ADX (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Williams fractals on 12h
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values,
    )
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate ADX on 12h for trend strength
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value is NaN
    
    # +DM and -DM
    up_move = np.diff(high_12h)
    down_move = -np.diff(low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = WilderSmooth(tr, period)
    plus_dm_smooth = WilderSmooth(plus_dm, period)
    minus_dm_smooth = WilderSmooth(minus_dm, period)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = WilderSmooth(dx, period)
    
    # Align indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bearish fractal AND volume spike AND ADX > 25 (strong trend)
            long_entry = (curr_close > bear_fract) and vol_spike and (adx_val > 25)
            # Short: price breaks below bullish fractal AND volume spike AND ADX > 25 (strong trend)
            short_entry = (curr_close < bull_fract) and vol_spike and (adx_val > 25)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below bullish fractal OR ADX < 20 (weakening trend)
            if (curr_close < bull_fract) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above bearish fractal OR ADX < 20 (weakening trend)
            if (curr_close > bear_fract) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0