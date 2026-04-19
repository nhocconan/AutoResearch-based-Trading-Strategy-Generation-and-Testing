#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13, 8 periods), Teeth (EMA8, 5 periods), Lips (EMA5, 3 periods)
# Trend: 1d EMA50 (long above, short below)
# Entry: Lips cross above Teeth (bullish) + volume + 1d uptrend (long)
#        Lips cross below Teeth (bearish) + volume + 1d downtrend (short)
# Exit: Opposite cross or 1.5x ATR stop
# Designed to capture trends in both bull and bear markets with minimal whipsaw
# Target: 15-25 trades/year to avoid fee drag
name = "12h_WilliamsAlligator_Trend_1d_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components on 12h timeframe
    # Jaw: EMA13 with 8 periods offset
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: EMA8 with 5 periods offset
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: EMA5 with 3 periods offset
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # 12h ATR for stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        # Lips-Teeth crossover signals
        lips_above_teeth = lips[i] > teeth[i]
        lips_below_teeth = lips[i] < teeth[i]
        lips_above_teeth_prev = lips[i-1] > teeth[i-1]
        lips_below_teeth_prev = lips[i-1] < teeth[i-1]
        
        bullish_cross = lips_above_teeth and not lips_above_teeth_prev
        bearish_cross = lips_below_teeth and not lips_below_teeth_prev
        
        if position == 0:
            # Long: Bullish cross + volume + 1d uptrend
            if bullish_cross and volume_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish cross + volume + 1d downtrend
            elif bearish_cross and volume_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish cross or 1.5x ATR stop
            if bearish_cross or close[i] < close[i-1] - 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish cross or 1.5x ATR stop
            if bullish_cross or close[i] > close[i-1] + 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals