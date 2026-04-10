#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + Weekly Regime Filter
# - Primary: 6h timeframe for balanced frequency and fee efficiency
# - HTF: 1w for trend regime (price vs weekly EMA200), 1d for volume confirmation
# - Long: Bull Power > 0 + Bear Power < 0 + price > weekly EMA200 + volume > 1.5x 20d MA
# - Short: Bull Power < 0 + Bear Power > 0 + price < weekly EMA200 + volume > 1.5x 20d MA
# - Exit: Opposite Elder Ray signal (Bull Power <= 0 for long exit, Bear Power <= 0 for short exit)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Works in bull/bear: Weekly EMA200 filter ensures we only trade with the major trend,
#   Elder Ray captures momentum within that trend, volume confirms conviction

name = "6h_1w_1d_elderray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly EMA200 for trend regime
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Elder Ray Power (6h)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Weekly trend regime: price above/below weekly EMA200
        uptrend = close_6h[i] > ema200_1w_aligned[i]
        downtrend = close_6h[i] < ema200_1w_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 (both positive momentum)
            # AND uptrend regime AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bull Power < 0 AND Bear Power > 0 (both negative momentum)
            # AND downtrend regime AND volume spike
            elif (bull_power[i] < 0 and bear_power[i] > 0 and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Elder Ray signal weakens (loss of momentum)
            if position == 1:  # Long position
                exit_condition = bull_power[i] <= 0  # Long momentum faded
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = bear_power[i] >= 0  # Short momentum faded
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals