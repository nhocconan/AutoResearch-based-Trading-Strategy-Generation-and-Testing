#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trending vs ranging markets
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 + volume > 1.5x 20 EMA
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 + volume > 1.5x 20 EMA
# Designed for 12h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Alligator filters out choppy markets, EMA50 ensures trend alignment, volume confirms momentum
# Works in bull markets (trend continuation signals) and bear markets (trend continuation signals)

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    
    def smma(values, period):
        """Smoothed Moving Average (Smma) - same as Wilder's smoothing"""
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple SMA
            result[period-1] = np.mean(values[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate SMMA for median price (typical price)
    typical_price = (high + low + close) / 3
    jaw = smma(typical_price, 13)
    teeth = smma(typical_price, 8)
    lips = smma(typical_price, 5)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First 8 values of jaw_shifted will be invalid due to roll
    jaw_shifted[:8] = np.nan
    # First 5 values of teeth_shifted will be invalid
    teeth_shifted[:5] = np.nan
    # First 3 values of lips_shifted will be invalid
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (no additional delay needed as SMMA is contemporaneous)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Calculate 12h volume EMA(20) for volume confirmation
    vol_12h = prices['volume'].values
    vol_series = pd.Series(vol_12h)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        # Alligator conditions
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator + price > 1d EMA50 + volume confirmation
            if (bullish_alignment and close[i] > ema_50_1d_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + price < 1d EMA50 + volume confirmation
            elif (bearish_alignment and close[i] < ema_50_1d_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish Alligator alignment OR price < 1d EMA50
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish Alligator alignment OR price > 1d EMA50
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals