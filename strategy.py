#!/usr/bin/env python3
name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 28:
        return np.zeros(n)
    
    # EMA13 for Elder Ray (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Daily EMA28 for trend filter (28-period EMA)
    ema28_1d = pd.Series(df_1d['close']).ewm(span=28, adjust=False, min_periods=28).mean().values
    ema28_1d_aligned = align_htf_to_ltf(prices, df_1d, ema28_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 28)
    
    for i in range(start_idx, n):
        if (np.isnan(ema28_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: daily EMA28 slope
        trend_up = ema28_1d_aligned[i] > ema28_1d_aligned[i-1]
        trend_down = ema28_1d_aligned[i] < ema28_1d_aligned[i-1]
        
        # Volume condition
        vol_spike = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + uptrend + volume spike
            if bull_power[i] > 0 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + downtrend + volume spike
            elif bear_power[i] < 0 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or trend changes
            if bull_power[i] <= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or trend changes
            if bear_power[i] >= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray with daily trend filter and volume confirmation
# - Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures buying/selling pressure
# - Long when Bull Power > 0 (buyers in control) + daily uptrend + volume spike
# - Short when Bear Power < 0 (sellers in control) + daily downtrend + volume spike
# - Daily EMA28 trend filter ensures alignment with higher timeframe trend
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buying power in uptrend) and bear (selling power in downtrend)
# - Exit when power shifts or trend changes
# - Position size 0.25 targets ~20-60 trades/year to avoid fee drag
# - Novel for 6m: Elder Ray not recently tried on 6h timeframe with 1d trend filter
# - Aims for 50-120 total trades over 4 years (12-30/year) to stay within limits
# - Elder Ray provides clear bull/bear power signals with zero lag when combined with EMA13
# - Daily trend filter reduces whipsaws vs same-timeframe signals
# - Volume confirmation reduces false signals during low-participation moves