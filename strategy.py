#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bull/Bear Power (Elder Ray) with 12h EMA trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Trend: 12h EMA34 (long above, short below)
# Entry: Strong Bull Power + rising Bull Power + volume + 12h uptrend (long)
#        Strong Bear Power + rising Bear Power + volume + 12h downtrend (short)
# Exit: Opposite power cross or 2x ATR stop
# Designed to work in both bull (strong bull power) and bear (strong bear power) markets
# Target: 25-40 trades/year to avoid fee drag
name = "4h_ElderRay_EMA12h_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Elder Ray: Bull Power and Bear Power using EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Smooth the power values for better signal
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # 4h ATR for stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or \
           np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        # Power rising/falling
        bull_power_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_power_rising = bear_power_smooth[i] > bear_power_smooth[i-1]
        
        if position == 0:
            # Long: Bull Power positive and rising + volume + 12h uptrend
            if bull_power_smooth[i] > 0 and bull_power_rising and volume_filter and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive and rising + volume + 12h downtrend
            elif bear_power_smooth[i] > 0 and bear_power_rising and volume_filter and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bear Power becomes positive (market weakening) or 2x ATR stop
            if bear_power_smooth[i] > 0 or close[i] < close[i-1] - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bull Power becomes positive (market strengthening) or 2x ATR stop
            if bull_power_smooth[i] > 0 or close[i] > close[i-1] + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals