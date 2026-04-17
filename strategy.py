#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bull/Bear Power (Elder Ray) with 1d EMA trend filter and volume confirmation.
# Enters long when Bull Power > 0 and Bear Power < 0 with rising Bull Power, short when opposite.
# Uses 1d EMA50 to filter trend direction (long only above, short only below).
# Requires 1d volume > 1.5x 20-day average to confirm institutional participation.
# Position size: 0.25. Target: 20-35 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA, volume and trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume and its 20-period average
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # 13-period EMA for Bull/Bear Power (Elder Ray)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > (1.5 * volume_ma20_1d_aligned[i])
        
        # Trend filter: price relative to 1d EMA50
        trend_filter_up = close[i] > ema50_1d_aligned[i]   # uptrend bias
        trend_filter_down = close[i] < ema50_1d_aligned[i] # downtrend bias
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, Bull Power rising, in uptrend
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                i > 13 and bull_power[i] > bull_power[i-1] and  # Bull Power rising
                volume_filter and 
                trend_filter_up):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0, Bull Power < 0, Bear Power falling, in downtrend
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  i > 13 and bear_power[i] < bear_power[i-1] and  # Bear Power falling (more negative)
                  volume_filter and 
                  trend_filter_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or trend filter fails
            if (bull_power[i] <= 0 or 
                not trend_filter_up or
                not volume_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive or trend filter fails
            if (bear_power[i] >= 0 or 
                not trend_filter_down or
                not volume_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ElderRay_1dEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0