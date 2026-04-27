#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_12hTrend_VolumeConfirm
Hypothesis: 6h strategy using Elder Ray Bull/Bear Power (EMA13) with 12h EMA34 trend filter and volume confirmation.
Enter long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 12h EMA34 (uptrend) AND volume > 1.5x 20-period average.
Enter short when Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND price < 12h EMA34 (downtrend) AND volume confirmation.
Exit on opposite Elder Ray signal (Bear Power > 0 for long exit, Bull Power < 0 for short exit) or 12h trend reversal.
Elder Ray captures momentum strength while filtering with higher timeframe trend reduces whipsaw in choppy markets.
Designed for low trade frequency (~12-25/year) with discrete position sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by following the 12h trend while using Elder Ray for precise momentum entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 6h timeframe (completed bars only)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 12h EMA34 (34) + 6h EMA13 (13) + volume avg (20)
    start_idx = max(34, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_12h_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Elder Ray momentum with 12h EMA34 trend filter and volume confirmation
            # Long: Bull Power > 0 (bullish momentum) AND Bear Power < 0 (confirming no bearish pressure) 
            #        AND price > EMA34 (12h uptrend) AND volume spike
            long_condition = (bull_val > 0) and (bear_val < 0) and (close_val > ema_val) and vol_conf
            # Short: Bear Power < 0 (bearish momentum) AND Bull Power > 0 (confirming no bullish pressure)
            #        AND price < EMA34 (12h downtrend) AND volume confirmation
            short_condition = (bear_val < 0) and (bull_val > 0) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Bear Power > 0 (bearish pressure appears) OR 12h EMA34 turns bearish (price below EMA)
            if (bear_val > 0) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bull Power < 0 (bullish pressure appears) OR 12h EMA34 turns bullish (price above EMA)
            if (bull_val < 0) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0