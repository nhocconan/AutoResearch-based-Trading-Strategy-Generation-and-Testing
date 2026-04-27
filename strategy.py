#!/usr/bin/env python3
"""
4h_ChaosBreakout_Volume_12hTrend
Hypothesis: Williams Fractal breakouts aligned with 12h EMA trend and volume spikes capture strong directional moves in both bull and bear markets. Fractals identify key support/resistance levels where price breaks through with momentum. Volume confirms institutional participation. Targets ~20-30 trades/year on 4h to minimize fee drag while maintaining edge in volatile markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n+1] and high[n-1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n+1] and low[n-1] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish = np.zeros(len(high_1d), dtype=bool)
    bullish = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high[i-2] < high[i-1] and 
            high[i] < high[i-1] and 
            high[i+1] < high[i-1] and 
            high[i+2] < high[i-1]):
            bearish[i] = True
        if (low[i-2] > low[i-1] and 
            low[i] > low[i-1] and 
            low[i+1] > low[i-1] and 
            low[i+2] > low[i-1]):
            bullish[i] = True
    
    # Convert to levels: bearish fractal = resistance, bullish fractal = support
    fractal_resistance = np.where(bearish, high_1d, np.nan)
    fractal_support = np.where(bullish, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    fractal_resistance = pd.Series(fractal_resistance).ffill().values
    fractal_support = pd.Series(fractal_support).ffill().values
    
    # Align fractal levels to 4h timeframe with 2-bar delay for confirmation
    # Williams fractals need 2 additional bars after the center bar for confirmation
    resistance_aligned = align_htf_to_ltf(prices, df_1d, fractal_resistance, additional_delay_bars=2)
    support_aligned = align_htf_to_ltf(prices, df_1d, fractal_support, additional_delay_bars=2)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        resistance = resistance_aligned[i]
        support = support_aligned[i]
        ema_trend = ema50_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above recent fractal resistance with uptrend and volume spike
            if close[i] > resistance and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below recent fractal support with downtrend and volume spike
            elif close[i] < support and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below recent fractal support or trend turns down
            if close[i] < support or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above recent fractal resistance or trend turns up
            if close[i] > resistance or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ChaosBreakout_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0