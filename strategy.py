#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA34 trend filter and volume confirmation
# Williams Fractals identify significant swing points where price reverses with confirmation
# Breakouts above/below recent fractal levels with volume indicate strong momentum
# 12h EMA34 ensures alignment with intermediate trend to reduce false signals
# Volume confirmation at 2.0x average filters low-participation moves
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag

name = "6h_WilliamsFractal_Breakout_12hEMA34_Volume"
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
    if len(df_12h) < 5:  # Need at least 5 bars for fractals
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Fractals: 5-bar pattern (2 left, 2 right)
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal = np.full(len(high_12h), np.nan)
    bullish_fractal = np.full(len(low_12h), np.nan)
    
    for i in range(2, len(high_12h) - 2):
        if (high_12h[i] > high_12h[i-2] and high_12h[i] > high_12h[i-1] and 
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            bearish_fractal[i] = high_12h[i]
        if (low_12h[i] < low_12h[i-2] and low_12h[i] < low_12h[i-1] and 
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            bullish_fractal[i] = low_12h[i]
    
    # Align 12h fractals to 6h timeframe with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price breaks above recent bearish fractal (resistance) AND price > 12h EMA34 AND volume spike
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below recent bullish fractal (support) AND price < 12h EMA34 AND volume spike
            elif (not np.isnan(bullish_fractal_aligned[i]) and 
                  close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below recent bullish fractal (support) OR closes below 12h EMA34 (trend change)
            if (not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i]) or \
               close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above recent bearish fractal (resistance) OR closes above 12h EMA34 (trend change)
            if (not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i]) or \
               close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals