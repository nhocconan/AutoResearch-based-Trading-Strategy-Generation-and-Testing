#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w EMA trend filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when: Bull Power > 0 AND Bear Power rising (less negative) AND 1w EMA34 up AND volume > 1.5x 20-period MA
# Short when: Bear Power < 0 AND Bull Power falling (less positive) AND 1w EMA34 down AND volume > 1.5x 20-period MA
# Exit when: Power signals reverse OR volume drops
# Uses Elder Ray for momentum conviction, weekly EMA for trend filter, volume for strength
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_1wEMA_VolumeConfirm"
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
    
    # Calculate EMA13 for Elder Ray on 6h
    if len(close) >= 13:
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema13 = np.full(n, np.nan)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Rising/Falling power (1-period change)
    bull_power_rising = np.concatenate([[False], bull_power[1:] > bull_power[:-1]])
    bear_power_falling = np.concatenate([[False], bear_power[1:] < bear_power[:-1]])
    bull_power_falling = np.concatenate([[False], bull_power[1:] < bull_power[:-1]])
    bear_power_rising = np.concatenate([[False], bear_power[1:] > bear_power[:-1]])
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 34:
        ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
        # Rising/falling EMA34 (trend direction)
        ema34_rising = np.concatenate([[False], ema34_1w[1:] > ema34_1w[:-1]])
        ema34_falling = np.concatenate([[False], ema34_1w[1:] < ema34_1w[:-1]])
    else:
        ema34_1w = np.full(len(close_1w), np.nan)
        ema34_rising = np.zeros(len(close_1w), dtype=bool)
        ema34_falling = np.zeros(len(close_1w), dtype=bool)
    
    # Align 1w EMA34 and trend to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema34_rising.astype(float))
    ema34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema34_falling.astype(float))
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_rising_aligned[i]) or 
            np.isnan(ema34_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bull Power rising AND 1w EMA34 up AND volume filter
            if (bull_power[i] > 0 and 
                bull_power_rising[i] and 
                ema34_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bear Power falling AND 1w EMA34 down AND volume filter
            elif (bear_power[i] < 0 and 
                  bear_power_falling[i] and 
                  ema34_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power >= 0 OR 1w EMA34 down
            if (bear_power[i] >= 0 or ema34_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power <= 0 OR 1w EMA34 up
            if (bull_power[i] <= 0 or ema34_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals