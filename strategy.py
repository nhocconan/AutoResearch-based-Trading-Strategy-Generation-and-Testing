#!/usr/bin/env python3
# 6h_ElderRay_BullPower_BearPower_1dTrend
# Hypothesis: Use Elder Ray indicator (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h,
# filtered by 1d EMA34 trend direction and volume confirmation. Enter long when Bull Power > 0 and rising,
# enter short when Bear Power < 0 and falling. Exit when power crosses zero or trend fails.
# Designed for low frequency (15-35 trades/year) to avoid fee drag. Works in bull (captures strength)
# and bear (captures weakness) with trend filter and volume confirmation.

name = "6h_ElderRay_BullPower_BearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(data, period):
    """Calculate EMA with proper handling of initial values."""
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray on 6h data: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = calculate_ema(close, 13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Daily EMA34 for trend filter
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Elder Ray signals: look for sustained power (not just instant)
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # LONG: Bull Power > 0 and rising, price above daily EMA34, volume confirmation
            if bull_power[i] > 0 and bull_power_rising and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 and falling, price below daily EMA34, volume confirmation
            elif bear_power[i] < 0 and bear_power_falling and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or trend fails
            if bull_power[i] <= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 or trend fails
            if bear_power[i] >= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals