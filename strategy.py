#!/usr/bin/env python3
# 12h_1w_donchian_breakout_volatility_regime_v1
# Hypothesis: Use 12h Donchian breakout with 1w ATR-based volatility regime filter and volume confirmation.
# In high volatility (ATR > median), breakouts are more likely to continue. In low volatility, avoid breakouts to reduce false signals.
# Works in bull markets (trend continuation) and bear markets (avoids breakdowns in low vol chop).
# Target: 10-25 trades/year per symbol (40-100 total over 4 years) by requiring volatility expansion and volume surge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_volatility_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1w data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 1w ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 1w ATR median for regime threshold
    atr_median = np.nanmedian(atr_1w[~np.isnan(atr_1w)])
    high_volatility = atr_1w_aligned > atr_median
    
    # Volume confirmation: volume > 1.5x average of last 12 periods (1.5 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 12h Donchian high with high volatility and volume
            if (close[i] > donchian_high_aligned[i] and 
                high_volatility[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 12h Donchian low with high volatility and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  high_volatility[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals