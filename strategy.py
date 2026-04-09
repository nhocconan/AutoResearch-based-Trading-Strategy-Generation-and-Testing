#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily ATR-based volatility regime with Camarilla pivot breakouts
# In low volatility regimes (ATR ratio < 0.8), price tends to revert to mean near Camarilla S3/R3
# In high volatility regimes (ATR ratio > 1.2), price tends to breakout and continue in direction of Camarilla S4/R4
# Volume confirmation filters false signals
# Works in bull/bear: adapts to volatility regime
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "6h_1d_camarilla_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for volatility regime
    atr_period = 14
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 1d ATR ratio (current vs 20-period average) for regime detection
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_20
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_r4 = close_1d + range_1d * 1.1 / 2.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    camarilla_s4 = close_1d - range_1d * 1.1 / 2.0
    
    # Align 1d indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit logic depends on volatility regime
            if atr_ratio_aligned[i] < 0.8:  # Low vol: mean reversion
                if close[i] < s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # High vol: trend continuation
                if close[i] < r3_aligned[i]:  # Exit if price falls below R3
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit logic depends on volatility regime
            if atr_ratio_aligned[i] < 0.8:  # Low vol: mean reversion
                if close[i] > r3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # High vol: trend continuation
                if close[i] > s3_aligned[i]:  # Exit if price rises above S3
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic depends on volatility regime
            if volume_confirmed:
                if atr_ratio_aligned[i] < 0.8:  # Low vol: mean reversion at S3/R3
                    if close[i] > s3_aligned[i] and close[i] < r3_aligned[i]:
                        # Range-bound: look for bounce off S3 or rejection at R3
                        if i > 0 and close[i-1] <= s3_aligned[i-1] and close[i] > s3_aligned[i]:
                            position = 1
                            signals[i] = 0.25
                        elif i > 0 and close[i-1] >= r3_aligned[i-1] and close[i] < r3_aligned[i]:
                            position = -1
                            signals[i] = -0.25
                else:  # High vol: breakout at S4/R4
                    if close[i] > r4_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < s4_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals