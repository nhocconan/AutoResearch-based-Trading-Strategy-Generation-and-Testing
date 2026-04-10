#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 12h regime filter and volume confirmation
# - Elder Ray Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND 12h close > EMA50 AND volume > 1.5x 20-bar avg
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND 12h close < EMA50 AND volume > 1.5x 20-bar avg
# - Exit when power signals reverse (Bull Power <= 0 for longs, Bear Power >= 0 for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA; regime filter ensures trend alignment
# - Works in bull markets (buy strength) and bear markets (sell weakness) with proper regime filter

name = "6h_12h_elder_ray_power_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute EMA13 for Elder Ray (12h timeframe)
    close_12h = df_12h['close'].values
    ema13_12h = pd.Series(close_12h).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Pre-compute EMA50 for regime filter (12h timeframe)
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * volume_20_avg_12h)
    
    # Align 12h indicators to 6h timeframe
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute Elder Ray Power components from 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Bull Power = High - EMA13
    bull_power_6h = high_6h - ema13_12h_aligned
    # Bear Power = EMA13 - Low
    bear_power_6h = ema13_12h_aligned - low_6h
    
    # Pre-compute power trends (rising/falling) using 2-bar difference
    bull_power_rising = np.zeros(n, dtype=bool)
    bear_power_rising = np.zeros(n, dtype=bool)
    bull_power_rising[2:] = bull_power_6h[2:] > bull_power_6h[:-2]
    bear_power_rising[2:] = bear_power_6h[2:] > bear_power_6h[:-2]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(bull_power_6h[i]) or
            np.isnan(bear_power_6h[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power rising AND regime bullish AND volume spike
            if (bull_power_6h[i] > 0 and 
                bear_power_rising[i] and 
                close_6h[i] > ema50_12h_aligned[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND Bull Power falling AND regime bearish AND volume spike
            elif (bear_power_6h[i] > 0 and  # Note: Bear Power is EMA-Low, so >0 means bearish
                  ~bull_power_rising[i] and  # Bull Power falling
                  close_6h[i] < ema50_12h_aligned[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power signals reverse
            # Exit long when Bull Power <= 0 (power gone)
            # Exit short when Bear Power <= 0 (selling pressure gone)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power_6h[i] <= 0:
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power_6h[i] <= 0:  # Bear Power weakening
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