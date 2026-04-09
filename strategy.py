#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) + 1w trend filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND weekly EMA34 trend is up (price > weekly EMA34) AND volume confirmation
# Short when Bear Power < 0 AND weekly EMA34 trend is down (price < weekly EMA34) AND volume confirmation
# Uses discrete position sizing 0.25 to target ~50-150 trades over 4 years (12-37/year)
# Works in bull/bear markets: follows the higher timeframe trend with Elder Ray as entry timing

name = "6h_1d_1w_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d Indicators (Elder Ray) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 1w Indicators (Trend Filter) ===
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend direction
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w indicators to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume Confirmation (6h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume (20-period)
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if Bear Power becomes negative (momentum fading)
            if bear_power_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Bull Power becomes positive (momentum fading)
            if bull_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Elder Ray alignment with weekly trend + volume confirmation
            weekly_uptrend = close[i] > ema34_1w_aligned[i]
            weekly_downtrend = close[i] < ema34_1w_aligned[i]
            
            if (bull_power_aligned[i] > 0 and weekly_uptrend and volume_confirmed):
                position = 1
                signals[i] = 0.25
            elif (bear_power_aligned[i] < 0 and weekly_downtrend and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals