#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combo with 1w EMA50 trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND Bear Power < 0 AND close > 1w EMA50 AND volume > 1.5x 20-period MA.
Short when Alligator jaws > teeth > lips (bearish alignment) AND Bull Power < 0 AND Bear Power > 0 AND close < 1w EMA50 AND volume > 1.5x 20-period MA.
Exit when Alligator alignment reverses or opposite Elder Ray condition hits.
Designed for ~15-25 trades/year with trend-following edge in both bull and bear markets.
Williams Alligator identifies trend structure; Elder Ray confirms bull/bear power; 1w EMA50 ensures higher timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator (13,8,5 SMAs with 8,5,3 shifts)
    # Jaw (blue): 13-period SMMA shifted 8 bars
    # Teeth (red): 8-period SMMA shifted 5 bars
    # Lips (green): 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the SMMA values
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from end
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # need EMA50, EMA13, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1w EMA50 = uptrend, close < 1w EMA50 = downtrend
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator alignment
        bullish_alignment = jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]
        bearish_alignment = jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]
        
        # Elder Ray conditions
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        bull_power_neg = bull_power[i] < 0
        bear_power_pos = bear_power[i] > 0
        
        if position == 0:
            # Long: Bullish alignment AND Bull Power > 0 AND Bear Power < 0 AND uptrend AND volume confirmation
            if bullish_alignment and bull_power_pos and bear_power_neg and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND Bull Power < 0 AND Bear Power > 0 AND downtrend AND volume confirmation
            elif bearish_alignment and bull_power_neg and bear_power_pos and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator alignment reverses or opposite Elder Ray condition
            exit_signal = False
            if position == 1:
                # Exit long if alignment turns bearish OR Bull Power <= 0 OR Bear Power >= 0
                exit_signal = (not bullish_alignment) or (bull_power[i] <= 0) or (bear_power[i] >= 0)
            elif position == -1:
                # Exit short if alignment turns bullish OR Bull Power >= 0 OR Bear Power <= 0
                exit_signal = (not bearish_alignment) or (bull_power[i] >= 0) or (bear_power[i] <= 0)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_ElderRay_1wEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0