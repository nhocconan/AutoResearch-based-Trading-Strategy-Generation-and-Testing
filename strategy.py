#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d Trend Filter and Volume Confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long when Bull Power > 0 and Bear Power rising (momentum) with 1d uptrend + volume spike
# - Short when Bear Power < 0 and Bull Power falling with 1d downtrend + volume spike
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Volume spike confirms momentum strength
# - Target: 12-30 trades/year (50-120 over 4 years) to minimize fee drag on 6h timeframe

name = "6h_ElderRay_1dTrend_Volume"
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
    
    # 1d data for trend filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d close for Elder Ray calculation
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power rising (improving) with 1d uptrend + volume spike
            bear_power_rising = bear_power_aligned[i] > bear_power_aligned[i-1]
            ema_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            long_cond = (bull_power_aligned[i] > 0 and 
                        bear_power_rising and
                        ema_rising and
                        volume_spike[i])
            
            # Short: Bear Power < 0 AND Bull Power falling (deteriorating) with 1d downtrend + volume spike
            bull_power_falling = bull_power_aligned[i] < bull_power_aligned[i-1]
            ema_falling = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
            short_cond = (bear_power_aligned[i] < 0 and 
                         bull_power_falling and
                         ema_falling and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power turns negative (momentum shift)
            if bear_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns positive (momentum shift)
            if bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals