#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above recent bullish fractal AND close > 1d EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below recent bearish fractal AND close < 1d EMA50 AND volume > 1.8x 20-period average.
Exit when price crosses opposite fractal level (bearish for long exit, bullish for short exit).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 30-60 trades/year per symbol.
Williams Fractals provide proven reversal/breakout edge by identifying key support/resistance levels.
1d EMA50 offers smooth trend filter for 4h timeframe alignment with reduced whipsaw.
Volume confirmation at 1.8x ensures only significant breakouts with participation are taken.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load 1d data for Williams Fractals - ONCE before loop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Compute Williams Fractals on 1d timeframe
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align fractals to 4h timeframe with extra 2-bar delay for confirmation
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal AND close > 1d EMA50 AND volume spike
            if (price > bullish_fractal_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal AND close < 1d EMA50 AND volume spike
            elif (price < bearish_fractal_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses opposite fractal level
            if position == 1 and price < bearish_fractal_aligned[i]:
                exit_signal = True
            elif position == -1 and price > bullish_fractal_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsFractal_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0