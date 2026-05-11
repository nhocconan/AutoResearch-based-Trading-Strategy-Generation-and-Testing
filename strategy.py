#5711
#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_Volume
Hypothesis: Use weekly Williams Fractal breakouts with 1w trend filter and volume confirmation to capture major breakouts in both bull and bear markets.
Trades only on confirmed weekly fractal breaks, ensuring low frequency and high signal quality. Uses Williams Fractal which requires 2-bar confirmation to avoid look-ahead.
"""

name = "1d_WilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams Fractals on weekly data (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    
    # Apply 2-bar additional delay for fractal confirmation (as per rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Calculate weekly EMA for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current daily volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly bullish fractal break AND price above weekly EMA20 AND volume filter
            if bullish_fractal_aligned[i] == 1.0 and close[i] > ema_20_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly bearish fractal break AND price below weekly EMA20 AND volume filter
            elif bearish_fractal_aligned[i] == 1.0 and close[i] < ema_20_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly bearish fractal break OR price below weekly EMA20
            if bearish_fractal_aligned[i] == 1.0 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: weekly bullish fractal break OR price above weekly EMA20
            if bullish_fractal_aligned[i] == 1.0 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals