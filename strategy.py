#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA trend filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (calculated on 6h)
- Trend filter: 1w EMA34 (long only when price > EMA34, short only when price < EMA34)
- Volume confirmation: > 1.8x 20-period average
- Entry: Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum) AND volume confirmation
          Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum) AND volume confirmation
- Exit: Opposite Elder Ray signal OR price crosses 1w EMA34
- Position size: 0.25 discrete level
- Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
- Works in bull/bear via 1w EMA trend filter + Elder Ray momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA13 for Elder Ray (on 6h close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Previous Elder Ray values for momentum check
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 34)  # Volume MA, EMA13, EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(bull_power_prev[i]) or
            np.isnan(bear_power_prev[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Elder Ray momentum conditions
        bull_momentum = bull_power[i] > 0 and bear_power[i] < bear_power_prev[i]  # Bullish momentum
        bear_momentum = bear_power[i] < 0 and bull_power[i] < bull_power_prev[i]  # Bearish momentum
        
        if position == 0:
            # Long: Bullish momentum AND price above 1w EMA34 AND volume confirmation
            if bull_momentum and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish momentum AND price below 1w EMA34 AND volume confirmation
            elif bear_momentum and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish momentum OR price crosses below 1w EMA34
            if bear_momentum or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish momentum OR price crosses above 1w EMA34
            if bull_momentum or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0