#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_elder_ray_regime_v1
# Uses Elder Ray (Bull/Bear Power) from 12h timeframe to determine regime and force direction.
# In bull regime (Bull Power > 0 and rising): only allow longs on 6h EMA(20) pullbacks to EMA(50).
# In bear regime (Bear Power < 0 and falling): only allow shorts on 6h EMA(20) rallies to EMA(50).
# Adds volume confirmation (volume > 1.5x 20-period average) to avoid low-quality signals.
# Designed for 6h timeframe with target 12-37 trades/year (50-150 over 4 years).
# Works in both bull and bear markets by adapting to the prevailing trend via Elder Ray.

name = "6h_12h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Elder Ray calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(13) and EMA(20) for Elder Ray
    close_12h = df_12h['close'].values
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA20
    bull_power = df_12h['high'].values - ema13_12h
    bear_power = df_12h['low'].values - ema20_12h
    
    # Align Elder Ray to 6h timeframe (1-bar delay for completed 12h bar)
    bull_power_6h = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # 6h EMA(20) and EMA(50) for pullback entries
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if Elder Ray not ready
        if np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]):
            signals[i] = 0.0
            continue
        
        # Determine regime from Elder Ray slope (using 3-period change)
        if i >= 3:
            bull_slope = bull_power_6h[i] - bull_power_6h[i-3]
            bear_slope = bear_power_6h[i] - bear_power_6h[i-3]
        else:
            bull_slope = 0
            bear_slope = 0
        
        # Bull regime: Bull Power > 0 and rising
        bull_regime = (bull_power_6h[i] > 0) and (bull_slope > 0)
        # Bear regime: Bear Power < 0 and falling
        bear_regime = (bear_power_6h[i] < 0) and (bear_slope < 0)
        
        # Skip if no clear regime
        if not (bull_regime or bear_regime):
            # Hold current position if exists
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check volume confirmation
        if not vol_confirm[i]:
            # Hold current position if exists
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Bull regime: look for longs on pullbacks to EMA(50)
        if bull_regime:
            # Long signal: price pulls back to EMA(50) from above and resumes up
            if (close[i] > ema50_6h[i] and 
                close[i-1] <= ema50_6h[i-1] and 
                close[i] > ema20_6h[i] and 
                position != 1):
                position = 1
                signals[i] = 0.25
            # Exit: price breaks below EMA(20)
            elif close[i] < ema20_6h[i] and position == 1:
                position = 0
                signals[i] = 0.0
            # Hold long
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
        
        # Bear regime: look for shorts on rallies to EMA(50)
        elif bear_regime:
            # Short signal: price rallies to EMA(50) from below and resumes down
            if (close[i] < ema50_6h[i] and 
                close[i-1] >= ema50_6h[i-1] and 
                close[i] < ema20_6h[i] and 
                position != -1):
                position = -1
                signals[i] = -0.25
            # Exit: price breaks above EMA(20)
            elif close[i] > ema20_6h[i] and position == -1:
                position = 0
                signals[i] = 0.0
            # Hold short
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals