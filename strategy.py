#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day EMA13 trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = Low - EMA13. 
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with EMA13 trending up and volume > 1.5x average.
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising, with EMA13 trending down and volume > 1.5x average.
# Uses 1-day EMA13 for trend filter to avoid counter-trend trades. Volume spike confirms conviction.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA13 for trend filter
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate 6th Elder Ray components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray (6-period timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: 6h volume > 1.5 x 20-period average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend: rising EMA13 = bullish, falling EMA13 = bearish
        daily_trend_up = ema13_1d_aligned[i] > ema13_1d_aligned[i-1] if i > 0 else False
        daily_trend_down = ema13_1d_aligned[i] < ema13_1d_aligned[i-1] if i > 0 else False
        
        # Elder Ray slope (1-period change)
        bull_power_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bull_power_falling = bull_power[i] < bull_power[i-1] if i > 0 else False
        bear_power_rising = bear_power[i] > bear_power[i-1] if i > 0 else False
        bear_power_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0 and falling, daily trend up, volume spike
            if (bull_power[i] > 0 and bull_power_rising and 
                bear_power[i] < 0 and bear_power_falling and
                daily_trend_up and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power < 0 and rising, daily trend down, volume spike
            elif (bear_power[i] < 0 and bear_power_falling and 
                  bull_power[i] < 0 and bull_power_rising and
                  daily_trend_down and volume_spike[i]):
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
            # Short exit: Bear Power turns positive or Bull Power turns negative
            if bear_power[i] >= 0 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA13Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0