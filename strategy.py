#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w EMA trend filter and volume confirmation.
# Uses weekly EMA for trend direction, daily Williams Fractals for breakout signals,
# and volume surge for confirmation. Works in bull (breakouts above bullish fractal) 
# and bear (breakdowns below bearish fractal). Target: 10-25 trades/year to avoid fee drag.
name = "1d_WilliamsFractal_1wEMA_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 21-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to daily timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_high = len(high)
    n_low = len(low)
    
    bearish_fractal = np.zeros(n_high, dtype=bool)
    bullish_fractal = np.zeros(n_low, dtype=bool)
    
    # Need at least 5 points for fractal pattern (2 on each side)
    for i in range(2, n_high - 2):
        if (high[i-2] < high[i-1] and 
            high[i] < high[i-1] and 
            high[i-1] > high[i-3] and 
            high[i-1] > high[i+1]):
            bearish_fractal[i] = True
    
    for i in range(2, n_low - 2):
        if (low[i-2] > low[i-1] and 
            low[i] > low[i-1] and 
            low[i-1] < low[i-3] and 
            low[i-1] < low[i+1]):
            bullish_fractal[i] = True
    
    # Williams fractals need 2 extra bars for confirmation (as per rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, pd.DataFrame({'high': high, 'low': low}), 
        bearish_fractal.astype(float), additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, pd.DataFrame({'high': high, 'low': low}), 
        bullish_fractal.astype(float), additional_delay_bars=2
    )
    
    # Volume confirmation: volume > 1.5x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 2  # Need at least 2 bars for fractal calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above bearish fractal level + 1w EMA > price (uptrend) + volume spike
            if (bearish_fractal_aligned[i] > 0 and 
                price > high[i-1] and  # Break above the fractal high
                ema_21_1w_aligned[i] > price and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below bullish fractal level + 1w EMA < price (downtrend) + volume spike
            elif (bullish_fractal_aligned[i] > 0 and 
                  price < low[i-1] and  # Break below the fractal low
                  ema_21_1w_aligned[i] < price and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below the fractal level or EMA flips down
            if price < high[i-1] or ema_21_1w_aligned[i] < price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above the fractal level or EMA flips up
            if price > low[i-1] or ema_21_1w_aligned[i] > price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals