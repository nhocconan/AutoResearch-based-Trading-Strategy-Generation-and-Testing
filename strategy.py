#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Fractal Breakout with weekly EMA trend filter and volume confirmation.
Trades breakouts of weekly Williams Fractals (significant support/resistance) in the direction of the weekly EMA trend.
Uses volume spike to confirm institutional interest at breakout. Designed for low trade frequency
(10-30 trades/year) to minimize fee flood and work in both bull and bear markets by aligning with
higher timeframe trend and using breakout logic rather than mean-reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low) fractals.
    Bearish fractal: middle bar has highest high of 5-bar window.
    Bullish fractal: middle bar has lowest low of 5-bar window.
    Returns arrays of same length with values where fractal exists, else NaN.
    """
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: current high is highest of 5 bars
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish[i] = high[i]
        # Bullish fractal: current low is lowest of 5 bars
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
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
    
    # Load weekly data for trend filter and fractal calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Williams Fractals
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1w, low_1w)
    # Williams fractals need 2 extra weekly bars for confirmation (after the center bar)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above bearish fractal (resistance) with uptrend bias
            if not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) with downtrend bias
            elif not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite fractal or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below bullish fractal (support) or closes below weekly EMA
                if (not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i]) or close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above bearish fractal (resistance) or closes above weekly EMA
                if (not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i]) or close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_1wEMA34_Volume"
timeframe = "6h"
leverage = 1.0