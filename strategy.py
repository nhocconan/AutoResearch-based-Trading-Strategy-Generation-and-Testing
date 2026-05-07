#!/usr/bin/env python3
name = "6h_1d_ElderRay_BullPower_BearPower_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(13) for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6-day EMA(8) for trend filter on 1d data (slower trend)
    ema8_1d = pd.Series(df_1d['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema8_1d_aligned = align_htf_to_ltf(prices, df_1d, ema8_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 4)  # Wait for Elder Ray and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema8_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) + price above EMA8 + volume
            bull_condition = bull_power_aligned[i] > 0
            uptrend = close[i] > ema8_1d_aligned[i]
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            
            if bull_condition and uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling) + price below EMA8 + volume
            elif bear_power_aligned[i] < 0 and not uptrend and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or volume drops
            if bull_power_aligned[i] <= 0 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or volume drops
            if bear_power_aligned[i] >= 0 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with daily trend and volume confirmation
# - Elder Ray measures bull/bear power relative to EMA13: bull_power = high - EMA13, bear_power = low - EMA13
# - Bull Power > 0 indicates buyers are stronger than average; Bear Power < 0 indicates sellers stronger
# - Combined with daily EMA8 trend filter to avoid counter-trend trades
# - Volume confirmation (1.5x average) ensures institutional participation
# - Works in bull markets (buy on Bull Power > 0 in uptrend) and bear markets (short on Bear Power < 0 in downtrend)
# - Exit when power signal reverses or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Novel application: Elder Ray on daily timeframe with 6h execution (not recently tried)
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Uses actual daily Elder Ray values (not 6h) for better signal quality