#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 1-day Williams Fractal reversal signals, filtered by 1-day EMA trend and volume spike confirmation.
# In trending markets, price often reverses at fractal support/resistance levels. We use bearish fractals (sell signals) in downtrends
# and bullish fractals (buy signals) in uptrends, confirmed by volume spikes to avoid false breakouts.
# This strategy targets reversals in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_WilliamsFractal_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: using align_ltf_to_htf as per actual function name in module

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close']
    ema_34 = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_ltf_to_htf(prices, df_1d, ema_34)
    
    # Calculate Williams Fractals on 1D data
    # Bearish fractal: high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    # Need at least 5 points for fractal calculation (2 on each side)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i] and 
            high_1d[i-1] < high_1d[i] and 
            high_1d[i+1] < high_1d[i] and 
            high_1d[i+2] < high_1d[i]):
            bearish_fractal[i] = True
            
        if (low_1d[i-2] > low_1d[i] and 
            low_1d[i-1] > low_1d[i] and 
            low_1d[i+1] > low_1d[i] and 
            low_1d[i+2] > low_1d[i]):
            bullish_fractal[i] = True
    
    # Williams fractals need 2 extra bars for confirmation (as per rules)
    bearish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish fractal + price above EMA34 (uptrend) + volume spike
            if bullish_fractal_aligned[i] and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal + price below EMA34 (downtrend) + volume spike
            elif bearish_fractal_aligned[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34 or bearish fractal with volume spike
            if close[i] < ema_34_aligned[i] or (bearish_fractal_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34 or bullish fractal with volume spike
            if close[i] > ema_34_aligned[i] or (bullish_fractal_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals