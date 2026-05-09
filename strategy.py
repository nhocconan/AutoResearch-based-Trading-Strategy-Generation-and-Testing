#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation.
# Uses weekly EMA for trend direction, Williams Fractals for swing high/low breakouts,
# and volume surge for confirmation. Works in bull (buy fractal breaks above weekly EMA)
# and bear (sell fractal breaks below weekly EMA). Target: 50-150 total trades over 4 years.
name = "6h_WilliamsFractalBreakout_1wEMA_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 21-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate Williams Fractals on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: bearish (sell) fractal = high with 2 lower highs on each side
    # bullish (buy) fractal = low with 2 higher lows on each side
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # For fractals, we need 2 additional daily bars for confirmation (total 3-day delay)
    bearish_fractal_confirmed = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal_confirmed = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(len(high_1d)):
        if bearish_fractal[i] and i + 2 < len(high_1d):
            bearish_fractal_confirmed[i + 2] = True
        if bullish_fractal[i] and i + 2 < len(low_1d):
            bullish_fractal_confirmed[i + 2] = True
    
    # Align weekly EMA to 6h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Align confirmed fractals to 6h timeframe (with 3-day delay already built in)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed.astype(float))
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed.astype(float))
    
    # Volume confirmation: volume > 2.0x 50-period EMA (moderate threshold)
    vol_ema50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_confirm = volume > (2.0 * vol_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for volume EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_21_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: bullish fractal breakout + price above weekly EMA + volume spike
            if (bullish_fractal_aligned[i] > 0 and price > ema_21_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal breakout + price below weekly EMA + volume spike
            elif (bearish_fractal_aligned[i] > 0 and price < ema_21_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly EMA or bearish fractal appears
            if price < ema_21_aligned[i] or bearish_fractal_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly EMA or bullish fractal appears
            if price > ema_21_aligned[i] or bullish_fractal_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals