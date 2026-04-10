#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d regime filter
# - Primary: 6h timeframe for lower trade frequency and better signal quality
# - HTF: 1d for regime detection (Bull/Bear Power EMA trend)
# - Long: 6h Bull Power > 0 AND 1d Bear Power EMA < 0 (bullish regime)
# - Short: 6h Bear Power < 0 AND 1d Bull Power EMA > 0 (bearish regime)
# - Exit: Opposite signal appears or power crosses zero
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Elder Ray captures institutional buying/selling pressure
# - Target: 75-175 total trades over 4 years (19-44/year) - within 6h sweet spot

name = "6h_1d_elderray_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h Elder Ray Components
    # Bull Power = High - EMA(close, 13)
    # Bear Power = Low - EMA(close, 13)
    close_6h_series = pd.Series(close_6h)
    ema_13 = close_6h_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_6h = high_6h - ema_13
    bear_power_6h = low_6h - ema_13
    
    # Calculate 1d Elder Ray Components for regime
    close_1d_series = pd.Series(close_1d)
    ema_13_1d = close_1d_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Smooth 1d power with EMA for regime filter (more stable)
    bull_power_ema_1d = pd.Series(bull_power_1d).ewm(span=10, min_periods=10, adjust=False).mean().values
    bear_power_ema_1d = pd.Series(bear_power_1d).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Align 1d regime to 6h bars (wait for completed 1d bar)
    bull_power_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_ema_1d)
    bear_power_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(bull_power_ema_1d_aligned[i]) or 
            np.isnan(bear_power_ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: 6h Bull Power positive AND 1d Bear Power EMA negative (bullish regime)
            if (bull_power_6h[i] > 0 and bear_power_ema_1d_aligned[i] < 0):
                position = 1
                signals[i] = 0.25
            # Short entry: 6h Bear Power negative AND 1d Bull Power EMA positive (bearish regime)
            elif (bear_power_6h[i] < 0 and bull_power_ema_1d_aligned[i] > 0):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when power signals reverse or cross zero
            if position == 1:  # Long position
                exit_condition = (
                    bull_power_6h[i] <= 0 or  # Bull power turns negative
                    bear_power_6h[i] >= 0     # Bear power turns positive
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    bear_power_6h[i] >= 0 or  # Bear power turns positive
                    bull_power_6h[i] <= 0     # Bull power turns negative
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals