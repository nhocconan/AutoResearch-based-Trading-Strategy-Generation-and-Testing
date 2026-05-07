#!/usr/bin/env python3
# 12h_Panic_Long_With_Weekly_Trend_Filter
# Hypothesis: Enter long on 12h when weekly trend is bullish (price > weekly EMA200) and
# 12h price closes below 12h Bollinger Lower Band (20,2) + volume spike > 2x average.
# Exit when price closes above 12h Bollinger Middle Band (20).
# Designed to catch mean-reversion bounces in strong uptrends, works in bull (continuation)
# and bear (bear market rallies). Weekly trend filter avoids counter-trend trades.
# Target: 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.

name = "12h_Panic_Long_With_Weekly_Trend_Filter"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 12h Bollinger Bands (20,2)
    close_s = pd.Series(close)
    ma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    lower_band = ma_20 - 2 * std_20
    middle_band = ma_20
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = close_s.rolling(window=20, min_periods=20).mean().values  # using close for volume MA approximation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(ma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend + price below BB lower + volume spike
            weekly_uptrend = close[i] > ema_200_1w_aligned[i]
            bb_oversold = close[i] < lower_band[i]
            volume_spike = volume[i] > 2 * vol_ma[i]
            
            if weekly_uptrend and bb_oversold and volume_spike:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit: price closes above BB middle band
            if close[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals