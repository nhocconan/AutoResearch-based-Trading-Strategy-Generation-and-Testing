#!/usr/bin/env python3
name = "6h_Williams_Fractal_Trend_1d"
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
    
    # 1d data for Williams Fractals (requires 5 candles: 2 left, center, 2 right)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Williams fractals require 2 extra bars for confirmation (center + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA34 trend filter (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Ensure enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish fractal (support) + price above EMA34 + volume spike
            if (bullish_fractal_aligned[i] == 1 and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal (resistance) + price below EMA34 + volume spike
            elif (bearish_fractal_aligned[i] == 1 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below EMA34 or bearish fractal appears
            if close[i] < ema_34_1d_aligned[i] or bearish_fractal_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above EMA34 or bullish fractal appears
            if close[i] > ema_34_1d_aligned[i] or bullish_fractal_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals