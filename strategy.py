#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d pivot points for support/resistance and 1w momentum filter.
# Pivot points provide clear support/resistance levels that work in both trending and ranging markets.
# Long when price crosses above R1 pivot with bullish 1w momentum (price > 1w EMA50) and volume confirmation.
# Short when price crosses below S1 pivot with bearish 1w momentum (price < 1w EMA50) and volume confirmation.
# Exit when price crosses the daily pivot point (PP) or momentum weakens.
# Designed to capture meaningful moves from key levels while avoiding false signals.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points for each day
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    
    # Load 1w data ONCE for momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w for momentum filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to lower timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need EMA50 and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Momentum filter: price above/below 1w EMA50
        bullish_momentum = close[i] > ema_50_1w_aligned[i]
        bearish_momentum = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for pivot breaks
            # Long: price crosses above R1 with bullish momentum
            if (close[i] > r1_aligned[i] and 
                bullish_momentum and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below S1 with bearish momentum
            elif (close[i] < s1_aligned[i] and 
                  bearish_momentum and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below PP or momentum turns bearish
            if (close[i] < pp_aligned[i] or 
                not bullish_momentum):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above PP or momentum turns bullish
            if (close[i] > pp_aligned[i] or 
                not bearish_momentum):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dPivot_1wEMA50_Momentum_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0