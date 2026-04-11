#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_extreme_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # CCI(20) on daily
    tp_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    sma_tp_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - sma_tp_1d) / (0.015 * mad_1d)
    
    # Volume spike detection on 4h (volume > 2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily CCI to 4h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if np.isnan(cci_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        volume_current = volume[i]
        cci_val = cci_1d_aligned[i]
        
        # Volume confirmation: volume spike
        volume_spike = volume_current > 2.0 * vol_ma_20[i]
        
        # Extreme CCI signals
        long_signal = cci_val < -150 and volume_spike
        short_signal = cci_val > 150 and volume_spike
        
        # Exit: CCI returns to neutral zone (-50 to 50)
        exit_long = cci_val > -50
        exit_short = cci_val < 50
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h CCI extreme with volume confirmation and daily timeframe alignment.
# Enters long when daily CCI < -150 (extreme oversold) with volume spike (>2x 20-period avg).
# Enters short when daily CCI > 150 (extreme overbought) with volume spike.
# Exits when CCI returns to neutral zone (-50 to 50).
# Uses daily CCI to avoid look-ahead and ensure alignment with higher timeframe extremes.
# Volume confirmation ensures institutional participation in the move.
# Target: 20-40 trades per year to minimize fee fade while capturing extreme reversals.
# Works in both bull and bear markets by fading extreme momentum with volume validation.