#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA trend direction and volume average.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (measures buying/selling pressure).
- Entry: Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum) AND price > 12h EMA50 (uptrend) AND volume > 1.5 * 12h average volume.
         Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum) AND price < 12h EMA50 (downtrend) AND volume > 1.5 * 12h average volume.
- Exit: Opposite Elder Ray signal (change in power polarity).
- Signal size: 0.25 discrete to minimize fee drag.
- Elder Ray captures the underlying power behind moves, filtering weak breakouts.
- 12h EMA50 ensures we trade with the higher timeframe trend.
- Volume confirmation avoids low-conviction moves.
- Works in both bull and bear markets by adapting to the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 70:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema_50_12h = ema(df_12h['close'].values, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h volume average for confirmation
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate Elder Ray components (Bull/Bear Power) using 13-period EMA
    ema_13 = ema(close, 13)
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need 50 for EMA50, 20 for volume MA, 13 for EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Previous power values for momentum check
        prev_bull_power = bull_power[i-1]
        prev_bear_power = bear_power[i-1]
        
        # Exit conditions: opposite Elder Ray signal (change in power polarity)
        if position != 0:
            # Exit long: Bear Power becomes positive (bulls losing control)
            if position == 1:
                if bear_power[i] > 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power becomes negative (bears losing control)
            elif position == -1:
                if bull_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with trend and volume confirmation
        if position == 0:
            # Bullish conditions: Bull Power > 0 (buying pressure) AND Bear Power falling (momentum)
            bullish_momentum = bull_power[i] > 0 and bear_power[i] < prev_bear_power
            # Bearish conditions: Bear Power < 0 (selling pressure) AND Bull Power falling (momentum)
            bearish_momentum = bear_power[i] < 0 and bull_power[i] < prev_bull_power
            
            # Trend filter: price relative to 12h EMA50
            uptrend = curr_close > ema_50_12h_aligned[i]
            downtrend = curr_close < ema_50_12h_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
            
            if bullish_momentum and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif bearish_momentum and downtrend and volume_confirm:
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