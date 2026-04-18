#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + Volume Spike + 1d EMA Trend Filter
Hypothesis: Williams fractals identify potential turning points. A breakout above/below the most recent fractal with volume confirmation and 1d EMA trend filter captures momentum in both bull and bear markets. The 1d EMA ensures we only trade in the direction of the higher timeframe trend, reducing whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: highest high with two lower highs on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: lowest low with two higher lows on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams fractals on 1d
    bearish_fractal, bullish_fractal = calculate_williams_fractals(
        df_1d['high'].values, df_1d['low'].values
    )
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 confirming bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_val = ema_1d_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above most recent bullish fractal + above 1d EMA + volume spike
            if (not np.isnan(bullish_fractal_val) and 
                close[i] > bullish_fractal_val and 
                close[i] > ema_val and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below most recent bearish fractal + below 1d EMA + volume spike
            elif (not np.isnan(bearish_fractal_val) and 
                  close[i] < bearish_fractal_val and 
                  close[i] < ema_val and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA
            if close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA
            if close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_VolumeSpike_1dEMA"
timeframe = "4h"
leverage = 1.0