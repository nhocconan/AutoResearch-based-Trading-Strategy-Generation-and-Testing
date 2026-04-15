#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Width regime + 1d Donchian breakout + volume confirmation
# In low volatility regimes (BBW < 20th percentile), wait for 1d Donchian(20) breakout with volume spike.
# Uses Bollinger Band Width to identify ranging markets where breakouts are more reliable.
# Works in both bull and bear markets by filtering for low volatility breakouts.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-hour Bollinger Band Width (20, 2) for regime detection
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = ma + bb_std * bb_std_dev
    lower = ma - bb_std * bb_std_dev
    bb_width = ((upper - lower) / ma) * 100  # Percentage
    
    # Regime filter: low volatility when BBW < 20th percentile
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).rank(pct=True) * 100
    low_volatility = bb_width_percentile < 20
    
    # 1-day Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    
    # Breakout signals
    breakout_up = (high_1d > donchian_high.shift(1))  # Break above upper band
    breakout_down = (low_1d < donchian_low.shift(1))   # Break below lower band
    
    # Align to 4h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down.astype(float))
    
    # Volume confirmation: current > 2x median of last 24 bars (1 day at 1h)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=10).median()
    vol_threshold = 2.0 * vol_median
    volume_spike = volume > vol_threshold
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(low_volatility[i]) or 
            np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Low volatility + upward breakout + volume spike
        if (low_volatility[i] and 
            breakout_up_aligned[i] > 0.5 and 
            volume_spike[i]):
            signals[i] = 0.25
        
        # Short: Low volatility + downward breakout + volume spike
        elif (low_volatility[i] and 
              breakout_down_aligned[i] > 0.5 and 
              volume_spike[i]):
            signals[i] = -0.25
        
        # Exit: Volatility increases (regime change) or opposite breakout
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (not low_volatility[i] or breakout_down_aligned[i] > 0.5)) or
               (signals[i-1] == -0.25 and (not low_volatility[i] or breakout_up_aligned[i] > 0.5)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_BBW_Donchian1d_Volume"
timeframe = "4h"
leverage = 1.0