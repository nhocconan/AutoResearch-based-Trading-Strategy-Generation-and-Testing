#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 12h EMA50 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 12h EMA50 AND volume > 1.5x average
# Exit when Elder Ray signals reverse or trend fails.
# Uses 6h timeframe for balance of frequency and reliability, Elder Ray for bull/bear strength, 12h EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via sustained buying pressure, bear via distribution signals.

name = "6h_ElderRay_12hTrend_Volume_v1"
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
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA(13) on 6h close for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (1.5 * vol_ma_6h)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power rising (current > previous) AND price > 12h EMA50 AND volume confirmation
            if bull_power[i] > 0 and bear_power[i] > bear_power[i-1] and close[i] > ema50_12h_aligned[i] and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bull Power falling (current < previous) AND price < 12h EMA50 AND volume confirmation
            elif bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and close[i] < ema50_12h_aligned[i] and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Elder Ray signals reverse OR trend fails (price < 12h EMA50)
            if bull_power[i] <= 0 or bear_power[i] >= bear_power[i-1] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Elder Ray signals reverse OR trend fails (price > 12h EMA50)
            if bear_power[i] >= 0 or bull_power[i] <= bull_power[i-1] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals