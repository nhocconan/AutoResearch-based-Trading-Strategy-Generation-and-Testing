#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility breakout with volume confirmation
# ATR expansion signals increasing volatility and potential trend initiation
# Volume confirmation filters low-conviction moves
# Works in bull/bear: volatility breakouts occur in both regimes, volume confirms validity
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "12h_1d_atr_breakout_volume_v1"
timeframe = "12h"
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
    
    # Calculate 1d ATR(14) for volatility measurement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio: current ATR / 20-period ATR average
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma_20
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility expansion: ATR ratio > 1.2 (increasing volatility)
        vol_expansion = atr_ratio_aligned[i] > 1.2
        
        if position == 1:  # Long position
            # Exit when volatility contracts (ATR ratio < 1.0) or mean reversion signal
            if atr_ratio_aligned[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when volatility contracts (ATR ratio < 1.0) or mean reversion signal
            if atr_ratio_aligned[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume and volatility confirmation
            # Long on volatility expansion + volume confirmation
            # Short on volatility expansion + volume confirmation (symmetrical)
            if vol_expansion and volume_confirmed:
                # Use price momentum direction for breakout bias
                price_momentum = close[i] - close[i-10]  # 10-bar momentum
                if price_momentum > 0:
                    position = 1
                    signals[i] = 0.25
                elif price_momentum < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals