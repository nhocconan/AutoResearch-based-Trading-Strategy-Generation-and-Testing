#!/usr/bin/env python3
# 1d_williams_alligator_1w_trend_volume_v1
# Hypothesis: Williams Alligator with 1-week trend filter and volume confirmation on daily timeframe.
# Long when price > Alligator teeth (middle line) and lips (green line) above teeth and teeth above jaws (red line)
# and 1-week EMA(50) rising and volume > 1.5x average.
# Short when price < Alligator teeth and lips below teeth and teeth below jaws and 1-week EMA(50) falling
# and volume > 1.5x average.
# Exit when price crosses back across the teeth or opposite signal.
# Williams Alligator uses smoothed moving averages (SMMA) with specific periods:
# Jaw (blue): 13-period SMMA, shifted 8 bars forward
# Teeth (red): 8-period SMMA, shifted 5 bars forward
# Lips (green): 5-period SMMA, shifted 3 bars forward
# This strategy aims to catch strong trends while avoiding choppy markets.
# Designed to work in both bull and bear markets by following the Alligator's alignment.
# Target: 15-25 trades/year to minimize fee drag while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_williams_alligator_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (using close prices)
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift the lines forward as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Handle the rolled values at the beginning
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Alligator alignment conditions
    # Bullish: Lips > Teeth > Jaw (all green above red above blue)
    bullish_alignment = (lips_shifted > teeth_shifted) & (teeth_shifted > jaw_shifted)
    # Bearish: Lips < Teeth < Jaw (all green below red below blue)
    bearish_alignment = (lips_shifted < teeth_shifted) & (teeth_shifted < jaw_shifted)
    
    # 1-week trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # 1-week EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1-week EMA slope (rising/falling)
    ema_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Volume confirmation: 1.5x average volume
    avg_volume = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_ok = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60  # Enough for Alligator and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_rising[i]) or np.isnan(ema_falling[i]) or \
           np.isnan(volume_ok[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below teeth or opposite signal
            if close[i] < teeth_shifted[i] or \
               (close[i] > jaw_shifted[i] and lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i] and ema_falling[i] and volume_ok[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above teeth or opposite signal
            if close[i] > teeth_shifted[i] or \
               (close[i] < jaw_shifted[i] and lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i] and ema_rising[i] and volume_ok[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish entry: price > teeth and bullish alignment and 1-week EMA rising and volume confirmation
            if close[i] > teeth_shifted[i] and bullish_alignment[i] and ema_rising[i] and volume_ok[i]:
                position = 1
                signals[i] = 0.25
            # Bearish entry: price < teeth and bearish alignment and 1-week EMA falling and volume confirmation
            elif close[i] < teeth_shifted[i] and bearish_alignment[i] and ema_falling[i] and volume_ok[i]:
                position = -1
                signals[i] = -0.25
    
    return signals