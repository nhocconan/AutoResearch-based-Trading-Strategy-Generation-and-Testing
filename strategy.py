#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray Index (Bull/Bear Power) with 1w EMA filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 1d timeframe
# - Long when Bull Power > 0 AND price > 1w EMA(50) AND volume > 1.5x 20-period average
# - Short when Bear Power < 0 AND price < 1w EMA(50) AND volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Elder Ray shows power of buyers/sellers, 1w EMA filters counter-trend trades
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1d_1w_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Bull Power and Bear Power
    bull_power = high_1d - ema_13_1d  # High - EMA(13)
    bear_power = low_1d - ema_13_1d   # Low - EMA(13)
    
    # Align Elder Ray components to 6h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA(50) to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: Bear Power turns negative OR price breaks below 1w EMA(50)
            if bear_power_aligned[i] < 0 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Bull Power turns positive OR price breaks above 1w EMA(50)
            if bull_power_aligned[i] > 0 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Elder Ray extreme + 1w EMA filter + volume confirmation
            if volume_confirmed:
                # Long entry: Bull Power positive AND price above 1w EMA(50)
                if bull_power_aligned[i] > 0 and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bear Power negative AND price below 1w EMA(50)
                elif bear_power_aligned[i] < 0 and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals