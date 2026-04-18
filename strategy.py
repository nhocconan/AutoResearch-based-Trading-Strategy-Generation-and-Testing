#!/usr/bin/env python3
"""
4h Williams Fractal Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Williams fractals identify key support/resistance levels. Breaking above a bearish fractal 
or below a bullish fractal with volume confirmation and 1d EMA trend filter captures momentum 
breakouts in both trending and ranging markets. Works in bull markets via upward breaks and 
in bear markets via downward breaks. Low trade frequency due to strict fractal confirmation.
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
    
    # Get 1d data for trend filter and fractals (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Fractals on 1d (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Bearish fractal: high[2] is highest of high[0:5] (needs 2 bars after)
    # Bullish fractal: low[2] is lowest of low[0:5] (needs 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_1d_aligned[i]
        vol_ok = vol_confirm[i]
        bearish_fractal_level = bearish_fractal_aligned[i]
        bullish_fractal_level = bullish_fractal_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above bearish fractal (resistance) with volume + uptrend
            if (not np.isnan(bearish_fractal_level) and 
                vol_ok and 
                close[i] > bearish_fractal_level and 
                close[i] > trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below bullish fractal (support) with volume + downtrend
            elif (not np.isnan(bullish_fractal_level) and 
                  vol_ok and 
                  close[i] < bullish_fractal_level and 
                  close[i] < trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below bullish fractal (support) or trend turns down
            if (not np.isnan(bullish_fractal_level) and close[i] < bullish_fractal_level) or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above bearish fractal (resistance) or trend turns up
            if (not np.isnan(bearish_fractal_level) and close[i] > bearish_fractal_level) or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0