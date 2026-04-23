#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray combo with 1d trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND 1d close > 1d EMA50 AND volume > 2.0x 20-period MA.
Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND 1d close < 1d EMA50 AND volume > 2.0x 20-period MA.
Exit when Alligator alignment breaks (jaws-teeth-lips not in order) OR Elder Ray power reverses sign.
Designed for low trade frequency (target: 20-50/year) with trend following in 4h timeframe.
Williams Alligator identifies trend phases, Elder Ray measures bull/bear power, daily trend filter ensures alignment with higher timeframe momentum.
Volume confirmation reduces false signals. Should work in both bull and bear markets by trading with the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs shifted)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13, 8, 5)  # need EMA50, volume MA20, Alligator, Elder Ray
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: jaws < teeth < lips = bullish, jaws > teeth > lips = bearish
        bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray power
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        
        # Trend filter: 1d close > EMA50 = uptrend, close < EMA50 = downtrend
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: bullish Alligator AND bull power > 0 AND uptrend AND volume filter
            if bullish_alignment and bull_power_pos and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND bear power < 0 AND downtrend AND volume filter
            elif bearish_alignment and bear_power_neg and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator alignment breaks OR Elder Ray power reverses
            exit_signal = False
            
            if position == 1:
                # Long exit: alignment breaks OR bull power <= 0
                if not bullish_alignment or bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: alignment breaks OR bear power >= 0
                if not bearish_alignment or bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0