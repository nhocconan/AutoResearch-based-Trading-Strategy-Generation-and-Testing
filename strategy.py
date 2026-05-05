#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA) AND Alligator Lips (8) > Teeth (13) > Jaw (13) AND 1d close > 1d EMA50 AND volume > 2.0x 20-period average
# Short when price < Alligator Jaw AND Alligator Lips < Teeth < Jaw AND 1d close < 1d EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses Alligator Jaw (trend reversal)
# Uses 12h primary timeframe with 1d HTF for trend filter and Alligator structure
# Williams Alligator uses smoothed moving averages (SMMA) with specific periods: Jaw=13, Teeth=8, Lips=5
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) based on proven Alligator performance
# Works in both bull and bear markets by following the 1d trend while using 12h for entry timing

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (Blue): 13-period SMMA of median price, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA of median price, shifted 5 bars forward
    # Lips (Green): 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set NaN for shifted values that don't have enough data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND Lips > Teeth > Jaw (bullish alignment) AND 1d close > 1d EMA50 AND volume spike
            if (close[i] > jaw_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND Lips < Teeth < Jaw (bearish alignment) AND 1d close < 1d EMA50 AND volume spike
            elif (close[i] < jaw_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Jaw (trend reversal)
            if close[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Jaw (trend reversal)
            if close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals