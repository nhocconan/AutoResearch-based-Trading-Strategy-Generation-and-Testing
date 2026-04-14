#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams Fractal levels + 1-week EMA trend filter.
# Long when price breaks above recent bearish fractal (resistance) with 1w EMA uptrend.
# Short when price breaks below recent bullish fractal (support) with 1w EMA downtrend.
# Williams Fractals identify swing points; breakouts from these levels with trend filter capture momentum.
# Exit on opposite fractal break or trend reversal. Designed for low-frequency, high-conviction trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n] > high[n-2], high[n] > high[n-1], high[n] > high[n+1], high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2], low[n] < low[n-1], low[n] < low[n+1], low[n] < low[n+2]
    n1d = len(high_1d)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)
    
    for i in range(2, n1d - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Forward fill to get most recent fractal level
    bearish_fractal = pd.Series(bearish_fractal).ffill().values
    bullish_fractal = pd.Series(bullish_fractal).ffill().values
    
    # Williams fractals need 2 extra bars for confirmation (after the center bar forms)
    bearish_fractal_confirm = np.full(n1d, np.nan)
    bullish_fractal_confirm = np.full(n1d, np.nan)
    bearish_fractal_confirm[2:] = bearish_fractal[:-2]
    bullish_fractal_confirm[2:] = bullish_fractal[:-2]
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slope = np.diff(ema_1w, prepend=np.nan)
    
    # Align indicators to 12h timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirm, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirm, additional_delay_bars=2)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    start = 30  # Need enough data for fractals and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(ema_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for fractal breakouts
            # Long: price breaks above bearish fractal (resistance) AND uptrend
            if (close[i] > bearish_fractal_aligned[i] and 
                ema_slope_aligned[i] > 0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below bullish fractal (support) AND downtrend
            elif (close[i] < bullish_fractal_aligned[i] and 
                  ema_slope_aligned[i] < 0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below bullish fractal (support) or trend reverses
            if (close[i] < bullish_fractal_aligned[i] or 
                ema_slope_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above bearish fractal (resistance) or trend reverses
            if (close[i] > bearish_fractal_aligned[i] or 
                ema_slope_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsFractal_1wEMA_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0