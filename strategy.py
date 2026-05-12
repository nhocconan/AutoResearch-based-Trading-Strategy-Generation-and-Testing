#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure.
# Long when Bull Power > 0 and Bear Power < 0 with price above 1d EMA34 and volume confirmation.
# Short when Bear Power > 0 and Bull Power < 0 with price below 1d EMA34 and volume confirmation.
# Exit when power signals weaken or reverse. Designed for 15-25 trades/year with institutional
# participation filters to avoid false signals. Works in bull via trend continuation and bear via
# reversals at institutional extremes. Uses 60-period EMA for power calculation and 1d EMA34 for trend.

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 60-period EMA for Elder Ray (standard period)
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema60  # Bull Power = High - EMA
    bear_power = low - ema60   # Bear Power = Low - EMA
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Align Elder Ray components to 6h timeframe (no additional delay needed)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_confirm = volume_confirm[i]
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_current = close_1d_aligned[i]
        
        if position == 0:
            # LONG: Bull Power > 0 (buying strength) and Bear Power < 0 (no selling pressure)
            # with price above 1d EMA34 and volume confirmation
            if bull_val > 0 and bear_val < 0 and close_1d_current > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (selling strength) and Bull Power < 0 (no buying pressure)
            # with price below 1d EMA34 and volume confirmation
            elif bear_val > 0 and bull_val < 0 and close_1d_current < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power becomes positive (selling pressure emerges) 
            # or Bull Power turns negative (buying weakness)
            if bear_val >= 0 or bull_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power becomes positive (buying pressure emerges)
            # or Bear Power turns negative (selling weakness)
            if bull_val >= 0 or bear_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals