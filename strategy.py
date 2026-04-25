#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams fractals identify swing points where price respects structure.
Breakouts above bearish fractals (resistance) or below bullish fractals (support)
with 1d EMA50 trend alignment and volume spike capture institutional moves in both bull and bear markets.
Discrete sizing (0.25) and ATR trailing stop (2.0x) minimize fee drag. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # 1d EMA50 trend filter (MTF) - loaded ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Fractals on 1d (MTF) - needs 2-bar confirmation delay
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar extra delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 50) + 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Williams fractal breakout conditions (use previous bar's levels to avoid look-ahead)
        breakout_long = curr_close > bearish_fractal_aligned[i-1]
        breakout_short = curr_close < bullish_fractal_aligned[i-1]
        
        if position == 0:
            # Look for entry signals - require: Fractal breakout + volume spike + 1d EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.0 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.0 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0