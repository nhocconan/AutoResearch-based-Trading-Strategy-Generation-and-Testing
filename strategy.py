#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d trend filter + volume confirmation
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND price > Alligator lips AND 1d bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when Alligator jaws > teeth > lips AND price < Alligator jaws AND 1d bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Uses Williams Alligator for trend identification and entry timing, 1d EMA50 for regime filter, volume confirmation to reduce false signals.
# Targets 20-40 trades/year on 4h timeframe for BTC/ETH/SOL with proper risk control via signal=0 exits.

name = "4h_WilliamsAlligator_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaws: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close_4h, 13)
    teeth = smma(close_4h, 8)
    lips = smma(close_4h, 5)
    
    # Shift as per Alligator definition
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that don't have enough history
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Alligator conditions: jaws < teeth < lips (bullish alignment) OR jaws > teeth > lips (bearish alignment)
    bullish_alligator = (jaws_shifted < teeth_shifted) & (teeth_shifted < lips_shifted)
    bearish_alligator = (jaws_shifted > teeth_shifted) & (teeth_shifted > lips_shifted)
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate volume spike filter (20-period volume EMA on 4h)
    vol_ema_20 = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_4h > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish Alligator alignment AND price > lips AND 1d bullish trend AND volume spike
            if (bullish_alligator[i] and 
                close_4h[i] > lips_shifted[i] and 
                trend_bullish_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Alligator alignment AND price < jaws AND 1d bearish trend AND volume spike
            elif (bearish_alligator[i] and 
                  close_4h[i] < jaws_shifted[i] and 
                  trend_bearish_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish Alligator alignment OR price < teeth OR 1d trend turns bearish
            if (bearish_alligator[i] or 
                close_4h[i] < teeth_shifted[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish Alligator alignment OR price > teeth OR 1d trend turns bullish
            if (bullish_alligator[i] or 
                close_4h[i] > teeth_shifted[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals