#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d volume confirmation and 1d trend filter
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA)
# - Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) + volume confirmation
# - Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) + volume confirmation
# - Volume filter: current 6h volume > 1.5x 20-period average 1d volume (scaled to 6h)
# - Trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50
# - Designed to capture momentum shifts with institutional participation (volume)
# - Works in bull markets (buying strength) and bear markets (selling weakness)
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "6h_ElderRay_1dVolume_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray Index: Bull Power and Bear Power (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 6h: 1d has 4x 6h bars, so divide by 4
        volume_factor = vol_ma_1d_aligned[i] / 4.0 if vol_ma_1d_aligned[i] > 0 else 0
        volume_filter = volume[i] > 1.5 * volume_factor
        
        if position == 0:
            # Look for long entry: bullish momentum (BP>0, BearP<0) + uptrend + volume
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish momentum (BearP>0, BP<0) + downtrend + volume
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when momentum turns bearish or trend breaks
            if bear_power[i] >= 0 or bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when momentum turns bullish or trend breaks
            if bull_power[i] >= 0 or bear_power[i] <= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals