#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Alligator's Jaw (blue line) AND 1d EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below Alligator's Jaw AND 1d EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit when price crosses back over Alligator's Teeth (red line) - mean reversion to equilibrium
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades in bear markets
# - Williams Alligator (SMMA-based) provides smooth trend identification with built-in filters
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)

name = "12h_1d_alligator_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator from 1d data
    # Alligator: Jaw (blue, 13-period SMMA, 8 bars forward), Teeth (red, 8-period SMMA, 5 bars forward), Lips (green, 5-period SMMA, 3 bars forward)
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(source, length):
        if length < 1:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is simple SMA
        if len(source) >= length:
            result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    jaw = smma(close_1d, 13)  # Jaw (blue)
    teeth = smma(close_1d, 8)  # Teeth (red)
    lips = smma(close_1d, 5)   # Lips (green)
    
    # Align Alligator lines to LTF (12h)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Jaw AND 1d uptrend with volume spike
            if (prices['close'].iloc[i] > jaw_aligned[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and  # price above 1d EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Jaw AND 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < jaw_aligned[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and  # price below 1d EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Teeth (mean reversion)
            # Exit when price crosses back over Teeth
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= teeth_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= teeth_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals