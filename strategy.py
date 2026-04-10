#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams Fractal Regime Filter
# - Primary: 6h timeframe for lower trade frequency (target: 12-37/year)
# - HTF: 1d for Elder Ray (bull/bear power) and Williams Fractals for regime
# - Long: 6h Elder Bull Power > 0 AND price > 6h EMA(20) AND 1d bullish fractal confirmed (with 2-bar delay)
# - Short: 6h Elder Bear Power < 0 AND price < 6h EMA(20) AND 1d bearish fractal confirmed (with 2-bar delay)
# - Exit: Elder Power crosses zero OR price reverts to 6h EMA(20)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Elder Ray shows underlying strength/weakness; fractals filter false breaks
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot

name = "6h_1d_elderray_fractal_v1"
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
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA(20) for trend filter
    close_6h_s = pd.Series(close_6h)
    ema_20_6h = close_6h_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13_6h = close_6h_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_6h = high_6h - ema_13_6h
    bear_power_6h = low_6h - ema_13_6h
    
    # Calculate 1d Williams Fractals
    # Bearish fractal: high[n-2] < high[n] AND high[n-1] < high[n] AND high[n+1] < high[n] AND high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] AND low[n-1] > low[n] AND low[n+1] > low[n] AND low[n+2] > low[n]
    n_1d = len(high_1d)
    bearish_fractal_1d = np.full(n_1d, np.nan)
    bullish_fractal_1d = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i] and high_1d[i-1] < high_1d[i] and 
            high_1d[i+1] < high_1d[i] and high_1d[i+2] < high_1d[i]):
            bearish_fractal_1d[i] = high_1d[i]
        if (low_1d[i-2] > low_1d[i] and low_1d[i-1] > low_1d[i] and 
            low_1d[i+1] > low_1d[i] and low_1d[i+2] > low_1d[i]):
            bullish_fractal_1d[i] = low_1d[i]
    
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_1d, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_1d, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_20_6h[i]) or np.isnan(bull_power_6h[i]) or 
            np.isnan(bear_power_6h[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND price > EMA(20) AND bullish fractal confirmed
            if (bull_power_6h[i] > 0 and close_6h[i] > ema_20_6h[i] and not np.isnan(bullish_fractal_aligned[i])):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND price < EMA(20) AND bearish fractal confirmed
            elif (bear_power_6h[i] < 0 and close_6h[i] < ema_20_6h[i] and not np.isnan(bearish_fractal_aligned[i])):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Elder Power crosses zero (loss of underlying strength/weakness)
            # 2. Price reverts to 6h EMA(20) (mean reversion)
            
            if position == 1:  # Long position
                exit_condition = (
                    bull_power_6h[i] <= 0 or  # Bull power crossed zero
                    close_6h[i] < ema_20_6h[i]  # Price reverted below EMA(20)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    bear_power_6h[i] >= 0 or  # Bear power crossed zero
                    close_6h[i] > ema_20_6h[i]  # Price reverted above EMA(20)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals