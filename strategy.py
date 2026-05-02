#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA50 trend filter and volume confirmation
# Williams Fractals identify potential reversal points with built-in confirmation (requires 2 subsequent bars)
# Breakouts above/below recent fractal levels with volume confirmation capture strong momentum
# 12h EMA50 ensures alignment with medium-term trend to reduce counter-trend trades
# Volume spike filter (2.0x average) ensures only high-participation moves are traded
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_WilliamsFractal_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Fractals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Williams Fractals: need 5 bars (pattern: low-low-HIGH-low-low for bearish, high-high-LOW-high-high for bullish)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    bearish_fractal = np.full(len(high_12h), np.nan)
    bullish_fractal = np.full(len(high_12h), np.nan)
    
    # Identify fractals (requires 2 bars after the center bar for confirmation)
    for i in range(2, len(high_12h) - 2):
        # Bearish fractal: high[i] is highest among 5 bars
        if (high_12h[i] > high_12h[i-2] and high_12h[i] > high_12h[i-1] and 
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            bearish_fractal[i] = high_12h[i]
        # Bullish fractal: low[i] is lowest among 5 bars
        if (low_12h[i] < low_12h[i-2] and low_12h[i] < low_12h[i-1] and 
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            bullish_fractal[i] = low_12h[i]
    
    # Align 12h fractals to 6h timeframe with additional delay for confirmation
    # Williams fractals need 2 extra 12h bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price breaks above bullish fractal AND price > 12h EMA50 AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below bearish fractal AND price < 12h EMA50 AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below bullish fractal (failed breakout) OR closes below 12h EMA50 (trend change)
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above bearish fractal (failed breakout) OR closes above 12h EMA50 (trend change)
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals