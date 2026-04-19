#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA)
# - 12h EMA34 trend filter: only take longs when price > 12h EMA34, shorts when price < 12h EMA34
# - Volume confirmation: current 6h volume > 1.5x 20-period average volume
# - Long when Bull Power > 0 and Bear Power < 0 (strong bullish momentum) + trend + volume
# - Short when Bear Power < 0 and Bull Power > 0 is false (strong bearish momentum) + trend + volume
# - Exit when power signals weaken or trend reverses
# - Designed to capture momentum in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drift

name = "6h_ElderRay_12hTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend direction
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: 20-period average volume
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: strong bullish momentum + uptrend + volume
            # Bull Power > 0 (strong buying pressure) and Bear Power < 0 (weak selling pressure)
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema34_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: strong bearish momentum + downtrend + volume
            # Bear Power < 0 (strong selling pressure) and Bull Power <= 0 (lack of buying pressure)
            elif bear_power[i] < 0 and bull_power[i] <= 0 and close[i] < ema34_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when bullish momentum weakens or trend reverses
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when bearish momentum weakens or trend reverses
            if bear_power[i] >= 0 or bull_power[i] > 0 or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals