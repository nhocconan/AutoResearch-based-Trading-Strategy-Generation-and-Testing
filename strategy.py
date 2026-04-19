#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d volume confirmation and 1w trend filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when Bull Power > 0 and Bear Power rising (less negative)
# - Short when Bear Power < 0 and Bull Power falling (less positive)
# - Volume filter: 6h volume > 1.5x 20-period average of 1d volume (scaled)
# - 1w EMA50 trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-35 trades/year to avoid excessive fee drag

name = "6h_ElderRay_1dVolume_1wTrend_v1"
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 6h: 1d has 4x 6h bars, so divide by 4
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 4.0)
        
        if position == 0:
            # Look for long entry: uptrend + Bull Power positive + Bear Power improving (less negative) + volume
            if (close[i] > ema_50_1w_aligned[i] and 
                bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative)
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend + Bear Power negative + Bull Power worsening (less positive) + volume
            elif (close[i] < ema_50_1w_aligned[i] and 
                  bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive)
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Bear Power turning negative or trend reversal
            if bear_power[i] < 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Bull Power turning positive or trend reversal
            if bull_power[i] > 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals