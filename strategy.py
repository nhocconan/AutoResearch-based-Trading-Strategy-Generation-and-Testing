#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Williams Alligator with 1-week trend filter, volume confirmation, and session filter
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction and entry points
# Jaw (13-period SMMA shifted 8 bars), Teeth (8-period SMMA shifted 5 bars), Lips (5-period SMMA shifted 3 bars)
# Long: Lips > Teeth > Jaw + price > Lips + volume spike + weekly uptrend
# Short: Lips < Teeth < Jaw + price < Lips + volume spike + weekly downtrend
# Target: 10-20 trades/year per symbol to avoid fee drag, works in bull/bear via weekly trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate SMMA (Smoothed Moving Average) - similar to RMA/Wilder's smoothing
    def smma(source, length):
        if length <= 0:
            return np.full_like(source, np.nan, dtype=np.float64)
        result = np.full_like(source, np.nan, dtype=np.float64)
        # First value is simple average
        if len(source) >= length:
            result[length-1] = np.mean(source[:length])
            # Subsequent values: (prev * (length-1) + current) / length
            for i in range(length, len(source)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    # Calculate Williams Alligator components on daily data
    lips = smma(close_1d, 5)   # 5-period SMMA
    teeth = smma(close_1d, 8)  # 8-period SMMA
    jaw = smma(close_1d, 13)   # 13-period SMMA
    
    # Shift the lines as per Alligator specification
    lips_shifted = np.roll(lips, 3)   # Shifted 3 bars forward
    teeth_shifted = np.roll(teeth, 5) # Shifted 5 bars forward
    jaw_shifted = np.roll(jaw, 8)     # Shifted 8 bars forward
    
    # Set initial values to NaN due to shifting
    lips_shifted[:3] = np.nan
    teeth_shifted[:5] = np.nan
    jaw_shifted[:8] = np.nan
    
    # Calculate weekly trend filter using EMA on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC (applied to daily data, so check if we're in session for the day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align all indicators to daily timeframe
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > Lips + volume spike + weekly uptrend
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > lips_aligned[i] and vol_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < Lips + volume spike + weekly downtrend
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < lips_aligned[i] and vol_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above Teeth (middle line)
            if position == 1:
                if close[i] < teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_Trend_Volume_Session"
timeframe = "1d"
leverage = 1.0