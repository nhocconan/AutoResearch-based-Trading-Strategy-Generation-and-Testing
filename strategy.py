#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Fractal Breakout with 1-day Trend Filter and Volume Confirmation.
Go long when price breaks above a bearish fractal (resistance) during 1-day uptrend with volume spike.
Go short when price breaks below a bullish fractal (support) during 1-day downtrend with volume spike.
Williams fractals identify key support/resistance levels that often hold in both trending and ranging markets.
The 1-day trend filter ensures we trade with the higher timeframe momentum, reducing whipsaws.
Volume confirmation adds conviction to breakouts.
Designed for low-to-moderate trade frequency suitable for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_fractals(high, low):
    """Calculate Williams fractals: bearish (resistance) and bullish (support)."""
    n = len(high)
    bearish = np.full(n, np.nan)  # resistance fractal
    bullish = np.full(n, np.nan)  # support fractal
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest among i-2, i-1, i, i+1, i+2
        if (high[i] >= high[i-1] and high[i] >= high[i-2] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest among i-2, i-1, i, i+1, i+2
        if (low[i] <= low[i-1] and low[i] <= low[i-2] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
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
    
    # Load 1-day data for trend and fractals - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Williams fractals on 1d high/low
    bearish_fractal, bullish_fractal = williams_fractals(
        df_1d['high'].values, df_1d['low'].values
    )
    # Fractals need 2 extra bars for confirmation (formation + 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) + 1d uptrend + volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                ema20_1d_aligned[i] > ema20_1d_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) + 1d downtrend + volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  ema20_1d_aligned[i] < ema20_1d_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to the opposite fractal level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below bullish fractal (support) or 1d trend turns down
                if (close[i] < bullish_fractal_aligned[i] or 
                    ema20_1d_aligned[i] < ema20_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above bearish fractal (resistance) or 1d trend turns up
                if (close[i] > bearish_fractal_aligned[i] or 
                    ema20_1d_aligned[i] > ema20_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0