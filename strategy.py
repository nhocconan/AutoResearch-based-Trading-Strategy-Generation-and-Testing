#!/usr/bin/env python3
# 6h_market_structure_bounce_v1
# Hypothesis: Mean reversion at key support/resistance levels with volume confirmation works across regimes.
# Uses 1-day swing high/low levels as dynamic support/resistance. Goes long when price touches
# 1-day low with bullish volume confirmation, short when price touches 1-day high with bearish volume.
# Includes 60-minute momentum filter to avoid catching falling knives. Designed for low trade frequency
# (~25-35 trades/year) to minimize fee impact in ranging and trending markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_market_structure_bounce_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1-day data ONCE before loop (critical for performance)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day swing points (high/low of each day)
    swing_high = df_1d['high'].values
    swing_low = df_1d['low'].values
    
    # Align to 6m timeframe - values update only after daily bar closes
    swing_high_6h = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_6h = align_htf_to_ltf(prices, df_1d, swing_low)
    
    # 60-period EMA for momentum filter (60 bars = 24 hours on 6m)
    close_series = pd.Series(prices['close'])
    ema_60 = close_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume confirmation: 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(60, n):  # Start after EMA warmup
        # Skip if any data is NaN
        if (np.isnan(swing_high_6h[i]) or np.isnan(swing_low_6h[i]) or 
            np.isnan(ema_60[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        close_price = prices['close'].iloc[i]
        
        # Long setup: price near 1-day low with bullish volume and above EMA
        near_support = abs(close_price - swing_low_6h[i]) / swing_low_6h[i] < 0.005  # Within 0.5%
        bullish_volume = volume[i] > vol_ma_20[i] * 1.3
        above_ema = close_price > ema_60[i]
        
        if near_support and bullish_volume and above_ema:
            signals[i] = 0.25
            
        # Short setup: price near 1-day high with bearish volume and below EMA
        elif (abs(close_price - swing_high_6h[i]) / swing_high_6h[i] < 0.005 and  # Within 0.5%
              volume[i] > vol_ma_20[i] * 1.3 and 
              close_price < ema_60[i]):
            signals[i] = -0.25
            
        # Flat otherwise
        else:
            signals[i] = 0.0
    
    return signals