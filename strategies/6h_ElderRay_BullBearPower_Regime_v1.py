#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v1
Hypothesis: Combine Elder Ray Bull/Bear Power with 12h trend regime filter. 
Bull Power = EMA(13) - Low; Bear Power = High - EMA(13). 
Long when Bull Power > 0 AND rising AND 12h close > EMA(50). 
Short when Bear Power > 0 AND rising AND 12h close < EMA(50). 
Volume confirmation required. Uses discrete sizing (0.25) to limit fee churn.
Target: 50-150 trades over 4 years = 12-37/year. Works in bull (trend continuation) and bear (counter-trend retracements) via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for regime filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Elder Ray components on 6h
    ema_period = 13
    ema_13 = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    bull_power = ema_13 - low  # Bull Power = EMA - Low
    bear_power = high - ema_13  # Bear Power = High - EMA
    
    # Slope of Bull/Bear Power (3-period change)
    bull_power_slope = bull_power - np.roll(bull_power, 3)
    bear_power_slope = bear_power - np.roll(bear_power, 3)
    # Handle roll NaNs
    bull_power_slope[:3] = 0
    bear_power_slope[:3] = 0
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(13), EMA(50) 12h, volume MA(20)
    start_idx = max(13, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        bull_slope = bull_power_slope[i]
        bear_slope = bear_power_slope[i]
        vol_conf = volume_confirm[i]
        regime_long = close_val > ema_50_12h_aligned[i]  # 12h uptrend
        regime_short = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: Bull Power positive AND rising AND volume confirm AND 12h uptrend
            long_signal = (bull_val > 0) and (bull_slope > 0) and vol_conf and regime_long
            
            # Short: Bear Power positive AND rising AND volume confirm AND 12h downtrend
            short_signal = (bear_val > 0) and (bear_slope > 0) and vol_conf and regime_short
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR 12h trend flips down
            if (bull_val <= 0) or (not regime_long):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power turns negative OR 12h trend flips up
            if (bear_val <= 0) or (not regime_short):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v1"
timeframe = "6h"
leverage = 1.0