#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike with 1w EMA50 trend filter.
- Williams Alligator (JAWS=13, TEETH=8, LIPS=5) identifies trend direction and alignment
- Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength
- Volume > 2.0x 20-period average for confirmation
- 1w EMA50 as higher timeframe trend filter (long only above, short only below)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted signals
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator: SMAs of median price (HL/2) with different periods
    median_price = (high + low) / 2.0
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Alligator's Jaw (13-period)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Alligator's Teeth (8-period)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Alligator's Lips (5-period)
    
    # Elder Ray: Bull Power and Bear Power using EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 50)  # Volume MA, Alligator jaws, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaws[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Alligator alignment: Lips > Teeth > Jaws (bullish) or Lips < Teeth < Jaws (bearish)
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaws[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaws[i]
        
        # Elder Ray confirmation: Bull Power > 0 and Bear Power < 0 for strong trends
        elder_bull_confirm = bull_power[i] > 0
        elder_bear_confirm = bear_power[i] < 0
        
        if position == 0:
            # Long: Alligator bullish alignment AND Elder Ray bullish confirmation AND price above 1w EMA50 AND volume confirmation
            if alligator_bullish and elder_bull_confirm and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment AND Elder Ray bearish confirmation AND price below 1w EMA50 AND volume confirmation
            elif alligator_bearish and elder_bear_confirm and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator bearish alignment OR price crosses below 1w EMA50
            if alligator_bearish or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator bullish alignment OR price crosses above 1w EMA50
            if alligator_bullish or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0