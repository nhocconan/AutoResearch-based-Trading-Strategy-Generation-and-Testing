#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12-hour VWAP deviation and volume confirmation.
# Uses deviation from VWAP as mean reversion signal, filtered by volume spike.
# Works in both bull and bear markets by adapting to volatility regimes and using volume as confirmation.
# Target: 20-50 trades per year to minimize fee drag while capturing mean reversion moves.

name = "4h_12h_vwap_deviation_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate VWAP for 12h period
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_numerator = (typical_price * df_12h['volume']).cumsum()
    vwap_denominator = df_12h['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap_values = vwap.values
    
    # Calculate deviation from VWAP as percentage
    deviation = (df_12h['close'] - vwap_values) / vwap_values * 100
    
    # Calculate 20-period standard deviation of deviation for z-score
    dev_series = pd.Series(deviation)
    dev_mean = dev_series.rolling(window=20, min_periods=20).mean().values
    dev_std = dev_series.rolling(window=20, min_periods=20).std().values
    z_score = np.where(dev_std != 0, (deviation - dev_mean) / dev_std, 0)
    
    # Calculate 20-period average volume for 12h
    vol_12h = df_12h['volume'].values
    vol_avg_20 = np.zeros_like(vol_12h, dtype=float)
    for i in range(19, len(vol_12h)):
        vol_avg_20[i] = np.mean(vol_12h[i-19:i+1])
    vol_avg_20[:19] = np.nan
    
    # Align 12h indicators to 4h timeframe
    z_score_aligned = align_htf_to_ltf(prices, df_12h, z_score)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if np.isnan(z_score_aligned[i]) or np.isnan(vol_avg_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 12h average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Mean reversion signals: extreme deviations from VWAP
        long_signal = z_score_aligned[i] < -2.0 and vol_filter
        short_signal = z_score_aligned[i] > 2.0 and vol_filter
        
        # Exit when deviation returns to zero (VWAP)
        exit_long = position == 1 and z_score_aligned[i] > -0.5
        exit_short = position == -1 and z_score_aligned[i] < 0.5
        
        # Priority: exit first, then entries
        if exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        elif long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals