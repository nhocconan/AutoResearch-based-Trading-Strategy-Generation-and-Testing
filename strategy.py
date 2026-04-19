#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Bullish: Bull Power > 0 AND Bear Power rising (less negative)
# - Bearish: Bear Power < 0 AND Bull Power falling (less positive)
# - 12h EMA34 trend filter: only take longs when price > 12h EMA34, shorts when price < 12h EMA34
# - Volume confirmation: current 6h volume > 1.5x 20-period average 6h volume
# - Designed to capture momentum in trending markets while filtering counter-trend moves
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "6h_ElderRay_12hTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Slope of Bear Power (for bullish confirmation: bear power rising = less negative)
    bear_power_series = pd.Series(bear_power)
    bear_power_slope = bear_power_series.diff().values
    
    # Slope of Bull Power (for bearish confirmation: bull power falling = less positive)
    bull_power_series = pd.Series(bull_power)
    bull_power_slope = bull_power_series.diff().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(bear_power_slope[i]) or \
           np.isnan(bull_power_slope[i]) or np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: bull power positive, bear power rising (less negative), uptrend, volume
            if bull_power[i] > 0 and bear_power_slope[i] > 0 and close[i] > ema34_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bear power negative, bull power falling (less positive), downtrend, volume
            elif bear_power[i] < 0 and bull_power_slope[i] < 0 and close[i] < ema34_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bear power turning negative or trend reversal
            if bear_power[i] < 0 or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bull power turning positive or trend reversal
            if bull_power[i] > 0 or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals