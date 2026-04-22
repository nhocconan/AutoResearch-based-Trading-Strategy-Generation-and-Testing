#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Fractal breakout with daily trend filter and volume confirmation.
Only trade breakouts of daily Williams Fractal levels in the direction of the daily EMA34 trend.
Williams Fractals identify key support/resistance levels that often act as breakout points.
Combined with daily trend filter to avoid counter-trend trades and volume confirmation to
ensure breakout validity. Designed for low trade frequency (12-25 trades/year) on 12h timeframe.
Works in both bull and bear markets by following the daily trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_fractals(high, low, n=2):
    """Calculate Williams Fractals: bearish (peak) and bullish (tower) patterns"""
    high = np.asarray(high)
    low = np.asarray(low)
    bearish = np.zeros(len(high), dtype=bool)
    bullish = np.zeros(len(low), dtype=bool)
    
    for i in range(n, len(high) - n):
        # Bearish fractal: highest high with n lower highs on each side
        if all(high[i] >= high[i - j] for j in range(1, n + 1)) and \
           all(high[i] >= high[i + j] for j in range(1, n + 1)):
            bearish[i] = True
        # Bullish fractal: lowest low with n higher lows on each side
        if all(low[i] <= low[i - j] for j in range(1, n + 1)) and \
           all(low[i] <= low[i + j] for j in range(1, n + 1)):
            bullish[i] = True
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Williams Fractals and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily
    bearish_fractal, bullish_fractal = williams_fractals(
        df_1d['high'].values, 
        df_1d['low'].values, 
        n=2
    )
    
    # Convert to price levels (0 where no fractal)
    bearish_levels = np.where(bearish_fractal, df_1d['high'].values, 0.0)
    bullish_levels = np.where(bullish_fractal, df_1d['low'].values, 0.0)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Williams Fractals need 2 extra daily bars for confirmation (pattern completes 2 bars after center)
    bearish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bearish_levels, additional_delay_bars=2
    )
    bullish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bullish_levels, additional_delay_bars=2
    )
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_confirmed[i]) or 
            np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0:
            # Long: Price breaks above bearish fractal (resistance) + daily uptrend + volume spike
            if (bearish_fractal_confirmed[i] > 0 and 
                close[i] > bearish_fractal_confirmed[i] and 
                ema34_aligned[i] > ema34_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal (support) + daily downtrend + volume spike
            elif (bullish_fractal_confirmed[i] > 0 and 
                  close[i] < bullish_fractal_confirmed[i] and 
                  ema34_aligned[i] < ema34_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite fractal level or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Price breaks below bullish fractal (support) or daily trend turns down
                if (bullish_fractal_confirmed[i] > 0 and 
                    close[i] < bullish_fractal_confirmed[i]) or \
                   ema34_aligned[i] < ema34_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price breaks above bearish fractal (resistance) or daily trend turns up
                if (bearish_fractal_confirmed[i] > 0 and 
                    close[i] > bearish_fractal_confirmed[i]) or \
                   ema34_aligned[i] > ema34_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0