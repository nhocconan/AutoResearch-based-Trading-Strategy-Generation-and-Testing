#!/usr/bin/env python3
"""
6h_ThreeSigma_Trend_With_1d_Volume_Spike
Hypothesis: Combines 6h price action relative to 20-period mean with 1d volume spikes and trend filters.
The strategy enters long when price is below mean - 2*std AND volume > 2*20-period average AND price > 6h EMA20.
Enters short when price is above mean + 2*std AND volume > 2*20-period average AND price < 6h EMA20.
This targets mean reversion during high-volume spikes, which often precede reversals in both bull and bear markets.
Uses 6h timeframe with 1d volume confirmation to reduce noise and increase edge.
Target: 15-35 trades per year per symbol, focusing on high-conviction setups.
"""

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
    
    # 6h EMA20 for trend filter
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6h Bollinger Bands (20, 2) for mean reversion signals
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    lower_band = sma20 - 2 * std20
    upper_band = sma20 + 2 * std20
    
    # 1d volume confirmation - get once before loop
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (2.0 * vol_avg_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = 40  # Need 20 for BB + 20 for vol
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema20_val = ema20[i]
        lower_val = lower_band[i]
        upper_val = upper_band[i]
        vol_spike_now = vol_spike_aligned[i]
        
        if position == 0:
            # Long: price at/below lower band, volume spike, above EMA20 (uptrend filter)
            if close_val <= lower_val and vol_spike_now and close_val > ema20_val:
                signals[i] = size
                position = 1
            # Short: price at/above upper band, volume spike, below EMA20 (downtrend filter)
            elif close_val >= upper_val and vol_spike_now and close_val < ema20_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses above EMA20 or reaches middle band
            if close_val >= ema20_val or close_val >= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses below EMA20 or reaches middle band
            if close_val <= ema20_val or close_val <= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ThreeSigma_Trend_With_1d_Volume_Spike"
timeframe = "6h"
leverage = 1.0