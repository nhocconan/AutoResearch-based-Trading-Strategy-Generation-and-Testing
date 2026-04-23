#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
Long when Bull Power > 0 AND price > 1d EMA34 AND volume > 1.5x 20-period MA.
Short when Bear Power < 0 AND price < 1d EMA34 AND volume > 1.5x 20-period MA.
Exit when Bull/Bear Power crosses zero or price crosses 1d EMA34 in opposite direction.
Uses 1d HTF for trend filter to align with major trend, Elder Ray measures bull/bear strength via EMA13,
volume confirms breakout conviction. Works in bull/bear by following higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # EMA13, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Bull Power > 0 AND price > 1d EMA34 AND volume filter
            if bull_val > 0 and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price < 1d EMA34 AND volume filter
            elif bear_val < 0 and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR price < 1d EMA34
                if bull_val <= 0 or price < ema_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power >= 0 OR price > 1d EMA34
                if bear_val >= 0 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0