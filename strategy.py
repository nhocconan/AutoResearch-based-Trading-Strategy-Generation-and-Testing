#!/usr/bin/env python3
# 6h Williams Fractal Breakout with Volume Spike and Daily Trend
# Hypothesis: Williams Fractals identify key reversal points. Breakouts above/below
# recent fractal levels with volume confirmation and daily EMA50 trend filter
# capture strong momentum moves. Works in both bull and bear markets by
# following breakout direction with strict entry criteria to limit trades.

name = "6h_WilliamsFractal_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # === Daily Data for Fractals and EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Williams Fractals (need 2-bar confirmation after center bar)
    bearish_fractal, bullish_fractal = compute_williams_fractals(daily_high, daily_low)
    
    # Additional delay of 2 bars for fractal confirmation (center + 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume Spike (20-period on 6h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above bullish fractal resistance + volume spike + price above daily EMA50
            if (bullish_fractal_aligned[i] > 0 and 
                close[i] > bullish_fractal_aligned[i] and
                vol_spike[i] and
                close[i] > ema_50_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bearish fractal support + volume spike + price below daily EMA50
            elif (bearish_fractal_aligned[i] > 0 and 
                  close[i] < bearish_fractal_aligned[i] and
                  vol_spike[i] and
                  close[i] < ema_50_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below nearest bearish fractal support
            if bearish_fractal_aligned[i] > 0 and close[i] < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above nearest bullish fractal resistance
            if bullish_fractal_aligned[i] > 0 and close[i] > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals