#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index with 12h volume confirmation and 1d trend filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 and Bear Power rising, short when Bear Power < 0 and Bull Power falling
# - Volume filter: current 4h volume > 1.5x 12h average volume (scaled)
# - Trend filter: only take longs when price > 1d EMA(50), shorts when price < 1d EMA(50)
# - Exit when Elder Ray signal reverses or trend fails
# - Designed to capture momentum with trend alignment, working in both bull and bear markets
# - Target: 25-40 trades/year to avoid excessive fee drag

name = "4h_ElderRay_12hVolume_1dTrend_v1"
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
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray Index (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 12h average volume (scaled)
        # Scale 12h average to 4h: 12h has 3x 4h bars, so divide by 3
        volume_filter = vol_ma_12h_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_12h_aligned[i] / 3.0)
        
        if position == 0:
            # Look for long entry: uptrend (price > 1d EMA50) + Bull Power > 0 + Bear Power rising + volume
            if (close[i] > ema_50_1d_aligned[i] and 
                bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1d EMA50) + Bear Power < 0 + Bull Power falling + volume
            elif (close[i] < ema_50_1d_aligned[i] and 
                  bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Bear Power >= 0 (bullish momentum fading) or trend failure
            if bear_power[i] >= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Bull Power <= 0 (bearish momentum fading) or trend failure
            if bull_power[i] <= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals