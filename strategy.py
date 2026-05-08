#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d Trend and Volume Filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Go long when Bull Power > 0 and Bear Power < 0 (bullish bias) with 1d uptrend
# - Go short when Bear Power > 0 and Bull Power < 0 (bearish bias) with 1d downtrend
# - Volume filter: require volume > 1.5x 20-period average to avoid low-conviction moves
# - Works in bull/bear by using 1d trend filter to align with higher timeframe momentum
# - Target: 20-40 trades/year to minimize fee drag on 6h timeframe

name = "6h_ElderRay_Power_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components: Bull Power and Bear Power
    # Using 13-period EMA as the reference (standard for Elder Ray)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = ema_13 - low   # EMA(13) - Low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish bias) + 1d uptrend + volume filter
            long_cond = (bull_power[i] > 0 and 
                        bear_power[i] < 0 and
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_filter[i])
            
            # Short: Bear Power > 0 AND Bull Power < 0 (bearish bias) + 1d downtrend + volume filter
            short_cond = (bear_power[i] > 0 and 
                         bull_power[i] < 0 and
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power becomes positive (momentum shift)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power becomes positive (momentum shift)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals