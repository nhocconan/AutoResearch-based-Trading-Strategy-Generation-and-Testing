#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (teeth) AND 1d bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when price < Alligator Jaw (teeth) AND 1d bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Uses Williams Alligator (Jaw=13-period SMMA, Teeth=8-period SMMA) to identify trend direction and avoid whipsaw
# Volume confirmation reduces false breakouts. Targets 12-37 trades/year on 12h timeframe.
# Works in bull markets via longs in bullish 1d trend regime and bear markets via shorts in bearish 1d trend regime.

name = "12h_WilliamsAlligator_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 12h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift Jaw and Teeth as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    # Lips not shifted for simplicity (using as reference)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup for Alligator calculations
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND price > Teeth AND 1d bullish trend AND volume spike
            # (Alligator alignment: Lips > Teeth > Jaw = bullish)
            if (close[i] > jaw_shifted[i] and 
                close[i] > teeth_shifted[i] and 
                lips[i] > teeth_shifted[i] and  # Lips above Teeth
                teeth_shifted[i] > jaw_shifted[i] and  # Teeth above Jaw
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND price < Teeth AND 1d bearish trend AND volume spike
            # (Alligator alignment: Lips < Teeth < Jaw = bearish)
            elif (close[i] < jaw_shifted[i] and 
                  close[i] < teeth_shifted[i] and 
                  lips[i] < teeth_shifted[i] and  # Lips below Teeth
                  teeth_shifted[i] < jaw_shifted[i] and  # Teeth below Jaw
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish (Lips < Teeth OR Teeth < Jaw) OR 1d trend turns bearish
            if (lips[i] < teeth_shifted[i] or 
                teeth_shifted[i] < jaw_shifted[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish (Lips > Teeth OR Teeth > Jaw) OR 1d trend turns bullish
            if (lips[i] > teeth_shifted[i] or 
                teeth_shifted[i] > jaw_shifted[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals