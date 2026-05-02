#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal Breakout with 1d EMA34 trend filter and volume confirmation
# Williams Fractals identify key swing points where price reverses or breaks out
# 1d EMA34 ensures trades only with intermediate-term trend, reducing false breakouts
# Volume confirmation at 2.0x average filters low-participation moves
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.30 to balance profit potential and fee drag
# Works in both bull (breakouts with trend) and bear (fades at fractal resistance/support)

name = "4h_WilliamsFractal_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals (5-bar: 2 left, center, 2 right)
    # Bearish fractal: high[i] is highest of [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish fractal breakout above recent high AND price > 1d EMA34 AND volume spike
            if (not np.isnan(bullish_fractal[i]) and 
                close[i] > bullish_fractal[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: Bearish fractal breakout below recent low AND price < 1d EMA34 AND volume spike
            elif (not np.isnan(bearish_fractal[i]) and 
                  close[i] < bearish_fractal[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below recent bullish fractal OR closes below 1d EMA34
            if (not np.isnan(bullish_fractal[i]) and close[i] < bullish_fractal[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price rises above recent bearish fractal OR closes above 1d EMA34
            if (not np.isnan(bearish_fractal[i]) and close[i] > bearish_fractal[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals