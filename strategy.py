#!/usr/bin/env python3
"""
4h_Stochastic_Pullback_1dTrend_Volume
Hypothesis: Stochastic oscillator identifies oversold/overbought conditions for pullback entries in trending markets. Combined with 1d EMA50 trend filter and volume confirmation to capture high-probability swing trades. Works in bull markets via long pullbacks and in bear markets via short rallies. Targets ~25-35 trades/year on 4h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Stochastic Oscillator (14,3,3) on 4h data
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    k_percent = 100 * ((close - lowest_low) / denominator)
    
    # Smooth %K to get %D (3-period SMA of %K)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Stochastic, EMA, and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(d_percent[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        k = k_percent[i]
        d = d_percent[i]
        ema_trend = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: Oversold bounce (K crosses above D from below 20) with uptrend and volume spike
            if k > d and k < 20 and d < 20 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: Overbought pullback (K crosses below D from above 80) with downtrend and volume spike
            elif k < d and k > 80 and d > 80 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: overbought condition or trend turns down
            if k > 80 or k < d or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: oversold condition or trend turns up
            if k < 20 or k > d or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Stochastic_Pullback_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0