#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1w EMA trend filter and volume confirmation
# This strategy uses Williams fractals to identify key support/resistance levels, entering
# on breakouts of bullish (buy) or bearish (sell) fractals with trend alignment from
# weekly EMA and volume confirmation. Works in both bull and bear markets by following
# the higher timeframe trend, with discrete position sizing to minimize transaction costs.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA trend (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=2)
    
    # Load daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (sell signal)
        if (high[i-2] < high[i-1] and 
            high[i-1] > high[i] and 
            high[i-3] < high[i-2] and 
            high[i+1] < high[i]):
            bearish_fractal[i] = True
        # Bullish fractal (buy signal)
        if (low[i-2] > low[i-1] and 
            low[i-1] < low[i] and 
            low[i-3] > low[i-2] and 
            low[i+1] > low[i]):
            bullish_fractal[i] = True
    
    # Store fractal values (price levels)
    bearish_fractal_val = np.where(bearish_fractal, high_1d, np.nan)
    bullish_fractal_val = np.where(bullish_fractal, low_1d, np.nan)
    
    # Align fractal levels to 4h timeframe with additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_val, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_val, additional_delay_bars=2)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to avoid index issues
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above bullish fractal + above 1w EMA + volume spike
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below bearish fractal + below 1w EMA + volume spike
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 1w EMA in opposite direction
            if position == 1:
                # Exit long: Close below 1w EMA
                if close[i] < ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Close above 1w EMA
                if close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1wEMA34_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0