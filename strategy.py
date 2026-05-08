#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h Trend Filter and Volume Confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when Bull Power > 0 and rising, Bear Power < 0, with 12h uptrend + volume spike
# - Short when Bear Power < 0 and falling, Bull Power < 0, with 12h downtrend + volume spike
# - Uses 12h EMA50 for trend filter to avoid counter-trend trades
# - Volume spike confirms institutional participation
# - Designed for 6h timeframe: targets 15-35 trades/year (60-140 total over 4 years)
# - Works in bull/bear markets by aligning with higher timeframe trend

name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive and rising, Bear Power negative, 12h uptrend + volume spike
            bull_rising = bull_power[i] > bull_power[i-1]
            long_cond = (bull_power[i] > 0 and 
                        bear_power[i] < 0 and
                        bull_rising and
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Bear Power negative and falling, Bull Power negative, 12h downtrend + volume spike
            bear_falling = bear_power[i] < bear_power[i-1]
            short_cond = (bear_power[i] < 0 and 
                         bull_power[i] < 0 and
                         bear_falling and
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or Bear Power turns positive
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive or Bull Power turns positive
            if bear_power[i] >= 0 or bull_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals