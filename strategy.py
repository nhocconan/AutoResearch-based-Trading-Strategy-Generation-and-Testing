#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# - Primary: 6h timeframe for balanced trade frequency and reduced fee drag
# - HTF: 1d for trend direction (EMA50) and volume regime
# - Long: Bull Power > 0 + Bear Power < 0 + close > EMA50(1d) + volume > 1.5x 20-period MA
# - Short: Bull Power < 0 + Bear Power > 0 + close < EMA50(1d) + volume > 1.5x 20-period MA
# - Exit: Opposite Elder Ray signal or close crosses EMA13(6h)
# - Position sizing: 0.25 (discrete level)
# - Target: 75-150 total trades over 4 years (19-37/year) - within 6h sweet spot
# - Works in bull/bear: Elder Ray measures bull/bear power relative to EMA13, effective in trending and ranging markets

name = "6h_1d_elderray_volume_v3"
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
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray (reference price)
    close_6h_s = pd.Series(close_6h)
    ema13_6h = close_6h_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components on 6h
    bull_power = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # Calculate 1d EMA50 for trend filter
    close_1d_s = pd.Series(close_1d)
    ema50_1d = close_1d_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 + close > EMA50(1d) + volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close_6h[i] > ema50_1d_aligned[i] and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bull Power < 0 AND Bear Power > 0 + close < EMA50(1d) + volume spike
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  close_6h[i] < ema50_1d_aligned[i] and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Opposite Elder Ray signal (change in market sentiment)
            # 2. Price crosses 6h EMA13 (trend change on lower timeframe)
            
            if position == 1:  # Long position
                exit_condition = (
                    (bull_power[i] < 0 and bear_power[i] > 0) or  # Opposite Elder Ray
                    close_6h[i] < ema13_6h[i]                     # Price below EMA13
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    (bull_power[i] > 0 and bear_power[i] < 0) or  # Opposite Elder Ray
                    close_6h[i] > ema13_6h[i]                     # Price above EMA13
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals