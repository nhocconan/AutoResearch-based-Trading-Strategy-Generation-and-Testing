#%%
#!/usr/bin/env python3
"""
6h_WilliamsFractal_Trend_Reversal
Hypothesis: On 6h timeframe, Williams Fractals identify swing points. 
In trending markets (ADX > 25), price often reverses after touching fractal levels.
Enter short at bearish fractal in uptrend, long at bullish fractal in downtrend.
Use 1d trend filter (EMA21) to avoid counter-trend trades. 
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
Works in both bull/bear via trend-following fractal reversals.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # 1d trend: bullish when close > EMA21
    bullish_trend = close_1d > ema21_1d
    bearish_trend = close_1d < ema21_1d
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(bullish_trend_aligned[i]) or 
            np.isnan(bearish_trend_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: fade fractals against trend
        short_signal = (bearish_fractal_aligned[i] > 0.5 and 
                       bullish_trend_aligned[i] > 0.5)
        long_signal = (bullish_fractal_aligned[i] > 0.5 and 
                      bearish_trend_aligned[i] > 0.5)
        
        # Exit when trend changes or opposite fractal appears
        trend_change_long = bearish_trend_aligned[i] > 0.5  # was bullish, now bearish
        trend_change_short = bullish_trend_aligned[i] > 0.5  # was bearish, now bullish
        
        if short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif trend_change_long and position == 1:
            signals[i] = -0.25  # Exit long
            position = 0
        elif trend_change_short and position == -1:
            signals[i] = 0.25   # Exit short
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_Trend_Reversal"
timeframe = "6h"
leverage = 1.0
#%%