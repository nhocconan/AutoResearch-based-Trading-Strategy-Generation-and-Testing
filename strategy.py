#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1-day trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Long when Bull Power > 0 AND daily EMA(34) > daily EMA(89) AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND daily EMA(34) < daily EMA(89) AND volume > 1.5x 20-period average.
# Exit when Bull Power <= 0 (for longs) or Bear Power <= 0 (for shorts).
# Elder Ray measures bull/bear strength relative to EMA, daily EMA crossover filters trend, volume confirms.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 90:
        return np.zeros(n)
    
    # Calculate daily EMA(34) and EMA(89) for trend
    close_d = df_d['close'].values
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_d = pd.Series(close_d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align daily EMAs to 6h timeframe
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    ema89_d_aligned = align_htf_to_ltf(prices, df_d, ema89_d)
    
    # Calculate EMA(13) for Elder Ray (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA(13)
    bear_power = ema13 - low   # EMA(13) - Low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 1)  # Sufficient warmup for EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_d_aligned[i]) or np.isnan(ema89_d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, daily EMA34 > EMA89 (uptrend), volume filter
            long_cond = (bull_power[i] > 0) and (ema34_d_aligned[i] > ema89_d_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power > 0, daily EMA34 < EMA89 (downtrend), volume filter
            short_cond = (bear_power[i] > 0) and (ema34_d_aligned[i] < ema89_d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (bullish momentum fading)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (bearish momentum fading)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals