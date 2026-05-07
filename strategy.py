#!/usr/bin/env python3
name = "6h_VolumeSpike_Reversal_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d Bollinger Bands for volatility regime and mean reversion signals
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume spike detection: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price near lower band with volume spike in uptrend
            if (close[i] <= lower_aligned[i] and vol_spike[i] and close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price near upper band with volume spike in downtrend
            elif (close[i] >= upper_aligned[i] and vol_spike[i] and close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses back above middle (SMA) or trend change
            if close[i] >= sma_20[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses back below middle (SMA) or trend change
            if close[i] <= sma_20[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Bollinger Band mean reversion with volume spike confirmation and 1d EMA(50) trend filter.
# In ranging markets, price tends to revert to the mean after reaching Bollinger Band extremes.
# Volume spikes at these extremes indicate exhaustion and higher probability of reversal.
# The 1d EMA(50) filter ensures trades align with the higher timeframe trend, reducing false signals.
# This strategy works in both bull and bear markets by adapting to the prevailing trend via the 1d EMA filter.
# Position size 0.25 limits drawdown while capturing mean reversion moves.
# Target: ~15-25 trades/year to avoid fee dust while capturing significant reversals.