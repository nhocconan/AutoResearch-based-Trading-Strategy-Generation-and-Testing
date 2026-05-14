#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA34 trend filter and volume spike confirmation
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > EMA34(12h) (uptrend) AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND price < EMA34(12h) (downtrend) AND volume > 1.5x 20-period average
# Exit when momentum diverges (Bull Power <= 0 for long, Bear Power >= 0 for short) OR trend flips
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Elder Ray measures bull/bear power via EMA(13), works in both bull (longs in uptrend+ bull power) and bear (shorts in downtrend+ bear power) markets.
# 12h EMA34 provides intermediate trend filter to avoid counter-trend whipsaws, volume spike confirms institutional participation.

name = "6h_ElderRay_BullBearPower_12hEMA34_Trend_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate Elder Ray Bull Power and Bear Power on 6h data
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > EMA34(12h) AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND price < EMA34(12h) AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (momentum lost) OR price < EMA34(12h) (trend flip)
            if (bull_power[i] <= 0 or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (momentum lost) OR price > EMA34(12h) (trend flip)
            if (bear_power[i] >= 0 or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals