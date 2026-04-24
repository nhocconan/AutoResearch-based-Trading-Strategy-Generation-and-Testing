#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend direction, 1d for Elder Ray calculation (based on daily high/low/close).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d data).
- Entry: Long when Bull Power > 0 AND Bear Power rising (improving) AND price > 12h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when Bear Power < 0 AND Bull Power falling (deteriorating) AND price < 12h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Elder Ray signal (Bear Power >= 0 for long exit, Bull Power <= 0 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Elder Ray measures bull/bear strength relative to EMA; rising/falling power indicates momentum persistence.
- Works in bull markets (strong Bull Power) and bear markets (strong Bear Power) with trend filter avoiding counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Elder Ray components (Bull Power, Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for EMA13
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    ema13_1d = ema(df_1d['close'].values, 13)
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Calculate Elder Ray momentum: rising/falling power
        # Bull Power rising: current > previous
        # Bear Power falling: current < previous (more negative)
        if i > 0:
            bull_power_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
            bear_power_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
        else:
            bull_power_rising = False
            bear_power_falling = False
        
        # Exit conditions: opposite Elder Ray signal
        if position != 0:
            # Exit long: Bear Power >= 0 (bullish momentum fading)
            if position == 1:
                if bear_power_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power <= 0 (bearish momentum fading)
            elif position == -1:
                if bull_power_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: Bull Power > 0 AND rising AND price > 12h EMA50
            long_condition = (bull_power_aligned[i] > 0 and 
                            bull_power_rising and 
                            curr_close > ema50_12h_aligned[i] and
                            volume_confirm)
            
            # Short: Bear Power < 0 AND falling AND price < 12h EMA50
            short_condition = (bear_power_aligned[i] < 0 and 
                             bear_power_falling and 
                             curr_close < ema50_12h_aligned[i] and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0