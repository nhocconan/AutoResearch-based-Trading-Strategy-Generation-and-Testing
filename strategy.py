#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume spike confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND close > EMA34(1d) AND volume > 2.0x 20-period average
# Short when Bear Power < 0 AND Bull Power > 0 AND close < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when Bull/Bear Power crosses zero OR EMA34(1d) trend flips
# Elder Ray measures bull/bear strength relative to EMA13, effective in both trending and ranging markets
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike (2.0x) ensures breakout has strong participation, reducing false signals
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (using close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Get 1d close for EMA34 calculation
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND close > EMA34(1d) AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power > 0 AND close < EMA34(1d) AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR close < EMA34(1d) (trend flip)
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power >= 0 OR Bear Power <= 0 OR close > EMA34(1d) (trend flip)
            if (bull_power[i] >= 0 or 
                bear_power[i] <= 0 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals