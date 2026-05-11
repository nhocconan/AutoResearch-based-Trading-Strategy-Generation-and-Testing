#!/usr/bin/env python3
name = "4h_WilliamsFractal_Reversal_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily data for Williams Fractal and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals (requires 5-bar window: 2 left, 2 right)
    # Bearish fractal: high[i] is highest of [i-2,i-1,i,i+1,i+2]
    # Bullish fractal: low[i] is lowest of [i-2,i-1,i,i+1,i+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Fractals need 2-bar confirmation after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA34 for trend filter (only needs completed daily bar)
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for fractals)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        bearish_fractal_signal = bearish_fractal_aligned[i]  # True when bearish fractal confirmed
        bullish_fractal_signal = bullish_fractal_aligned[i]   # True when bullish fractal confirmed
        
        if position == 0:
            # Long: Bullish fractal + above daily EMA34 + volume spike
            if bullish_fractal_signal and price_above_ema1d and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal + below daily EMA34 + volume spike
            elif bearish_fractal_signal and price_below_ema1d and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: fractal reversal or trend change
            if position == 1:
                # Exit long: Bearish fractal appears OR price crosses below EMA34
                if bearish_fractal_signal or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Bullish fractal appears OR price crosses above EMA34
                if bullish_fractal_signal or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals