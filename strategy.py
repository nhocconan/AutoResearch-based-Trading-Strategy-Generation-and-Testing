#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume confirmation
# - Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with volume confirmation
# - Trend filter: 12h EMA34 - only take longs when price > 12h EMA34, shorts when price < 12h EMA34
# - Volume: current 6h volume > 1.5x 20-period average of 12h volume (scaled)
# - Designed to capture institutional buying/selling pressure with trend alignment
# - Target: 20-35 trades/year to avoid excessive fee drag

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
    
    # Get 12h data for trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h volume average (20-period) scaled to 6h
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 12h average volume (scaled)
        # Scale 12h average to 6h: 12h has 2x 6h bars, so divide by 2
        volume_filter = vol_ma_12h_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_12h_aligned[i] / 2.0)
        
        if position == 0:
            # Look for long entry: Bull Power positive and rising, Bear Power negative and falling, uptrend, volume
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                close[i] > ema_34_12h_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: Bear Power negative and falling, Bull Power positive and rising, downtrend, volume
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                  close[i] < ema_34_12h_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Bull Power turns negative or trend reverses
            if bull_power[i] <= 0 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Bear Power turns positive or trend reverses
            if bear_power[i] >= 0 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals