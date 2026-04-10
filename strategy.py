#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d Volume Regime Filter
# - Primary: 6h timeframe for balanced trade frequency and reduced fee drag
# - HTF: 1d for volume confirmation and volatility regime (avoid low-volume chop)
# - Long: Bull Power > 0 (close > EMA13) AND Bear Power < 0 (high < EMA13) AND 1d volume > 1.5x 20-period MA
# - Short: Bear Power < 0 (high < EMA13) AND Bull Power < 0 (close < EMA13) AND 1d volume > 1.5x 20-period MA
# - Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 6h sweet spot
# - Works in bull/bear: Elder Ray captures trend strength via price-EMA relationship; volume filter ensures participation

name = "6h_1d_elderray_power_v1"
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
    
    # Calculate 6h EMA(13) for Elder Ray Power
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate 6h Elder Ray Power
    bull_power = close_6h - ema13_6h  # Close - EMA13
    bear_power = high_6h - ema13_6h   # High - EMA13 (note: Elder Ray uses high for bear power)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA (avoid low-volume false signals)
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 (price above EMA13 with rejection of higher highs) AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power < 0 (price below EMA13 with rejection of lower lows) AND volume spike
            elif (bear_power[i] < 0 and bull_power[i] < 0 and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Opposite Elder Ray signal
            if position == 1:  # Long position
                exit_condition = bull_power[i] < 0  # Bull Power turned negative
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = bear_power[i] > 0  # Bear Power turned positive
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals