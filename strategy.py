#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter and volume confirmation
# - Primary: 6h timeframe for balanced trade frequency and reduced fee drag
# - HTF: 12h for trend direction (EMA50) and 1d for volume confirmation
# - Long: 6h Bull Power > 0 AND 12h EMA50 uptrend (close > EMA50) AND 1d volume > 1.5x 20-period MA
# - Short: 6h Bear Power < 0 AND 12h EMA50 downtrend (close < EMA50) AND 1d volume > 1.5x 20-period MA
# - Exit: Opposite Elder Ray signal (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Works in bull/bear: Elder Ray captures bull/bear power, 12h EMA50 filters counter-trend trades, volume confirms conviction

name = "6h_12h_1d_elderray_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray (standard period)
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate 6h Bull Power and Bear Power
    bull_power_6h = high_6h - ema13_6h
    bear_power_6h = low_6h - ema13_6h
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: 12h EMA50 direction
        uptrend = close_12h[i] > ema50_12h_aligned[i]
        downtrend = close_12h[i] < ema50_12h_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (bulls in control) + uptrend + volume spike
            if (bull_power_6h[i] > 0 and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 (bears in control) + downtrend + volume spike
            elif (bear_power_6h[i] < 0 and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Elder Ray signal (loss of bull/bear power)
            if position == 1:  # Long position
                exit_condition = bull_power_6h[i] <= 0  # Bulls losing control
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = bear_power_6h[i] >= 0  # Bears losing control
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals