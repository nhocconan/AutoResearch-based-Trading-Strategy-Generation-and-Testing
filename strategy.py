#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume average.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Entry: Long when Bull Power > 0 AND Bear Power increasing (less negative) AND price > 1d EMA34 AND volume > 1.5 * 20-period 1d volume average.
         Short when Bear Power < 0 AND Bull Power decreasing (less positive) AND price < 1d EMA34 AND volume > 1.5 * 20-period 1d volume average.
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Elder Ray measures bull/bear strength relative to EMA. Works in both bull and bear markets by measuring power shifts.
- Volume confirmation ensures legitimacy of power shifts.
- 1d EMA34 filter ensures trading with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(series, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    if len(close) < 13:  # Need sufficient data for EMA13
        return np.zeros(n)
    
    ema13 = ema(close, 13)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # Need 13 for EMA13, 34 for 1d EMA34, 20 for 1d volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Elder Ray power changes (1-bar momentum)
        if i >= 1:
            bull_power_change = bull_power[i] - bull_power[i-1]
            bear_power_change = bear_power[i] - bear_power[i-1]
        else:
            bull_power_change = 0
            bear_power_change = 0
        
        # Exit conditions: opposite Elder Ray signal
        if position != 0:
            # Exit long: Bull Power becomes negative (bears taking control)
            if position == 1:
                if bull_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bear Power becomes positive (bulls taking control)
            elif position == -1:
                if bear_power[i] > 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with 1d EMA trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period 1d volume average
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # 1d EMA34 trend filter
            uptrend_1d = curr_close > ema34_1d_aligned[i]
            downtrend_1d = curr_close < ema34_1d_aligned[i]
            
            # Elder Ray conditions
            # Long: Bull Power > 0 (bulls in control) AND Bear Power increasing (less negative = bearish momentum weakening)
            long_condition = (bull_power[i] > 0) and (bear_power_change > 0) and uptrend_1d and volume_confirm
            
            # Short: Bear Power < 0 (bears in control) AND Bull Power decreasing (less positive = bullish momentum weakening)
            short_condition = (bear_power[i] < 0) and (bull_power_change < 0) and downtrend_1d and volume_confirm
            
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

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0