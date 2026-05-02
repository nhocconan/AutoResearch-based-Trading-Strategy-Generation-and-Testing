#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Williams Fractals on 6h timeframe provide structural support/resistance levels with higher frequency than 1d
# 1d EMA34 ensures trades only with intermediate-term trend, reducing false breakouts in choppy markets
# Volume confirmation at 1.5x average filters low-participation moves
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 to minimize fee churn
# Works in both bull and bear markets by following trend with structural breakouts

name = "6h_WilliamsFractal_Breakout_1dEMA34_Volume"
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
    
    # Calculate Williams Fractals on 6h timeframe (requires 5-bar window)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Bearish fractal: current high is highest of [n-2, n-1, n, n+1, n+2]
    # Bullish fractal: current low is lowest of [n-2, n-1, n, n+1, n+2]
    bearish_fractal = (high_series.rolling(window=5, center=True, min_periods=5).max() == high_series).values
    bullish_fractal = (low_series.rolling(window=5, center=True, min_periods=5).min() == low_series).values
    
    # 1d EMA34 for trend filter - loaded ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            i >= len(bearish_fractal) or i >= len(bullish_fractal)):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish fractal confirmed AND price > 1d EMA34 AND volume spike
            if (bullish_fractal[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal confirmed AND price < 1d EMA34 AND volume spike
            elif (bearish_fractal[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below 1d EMA34 OR bearish fractal forms
            if close[i] < ema_34_1d_aligned[i] or bearish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above 1d EMA34 OR bullish fractal forms
            if close[i] > ema_34_1d_aligned[i] or bullish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals