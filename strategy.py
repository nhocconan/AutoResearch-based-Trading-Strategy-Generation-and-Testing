#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend filter + volume confirmation
# Long when: price breaks above Donchian(20) high AND 1d EMA50 > EMA200 (bullish trend) AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) low AND 1d EMA50 < EMA200 (bearish trend) AND volume > 1.5x 20-period MA
# Exit when: price reverses to Donchian midpoint OR trend filter weakens
# Uses Donchian for structure, 1d EMA for higher timeframe trend, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fees.

name = "4h_Donchian20_1dEMA_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # need sufficient data for EMA200
        return np.zeros(n)
    
    # Calculate EMAs on 1d
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Trend filter: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1d EMAs to 4h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish.astype(float))
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish.astype(float))
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high + bullish 1d trend + volume filter
            if (close[i] > donchian_high[i] and 
                ema_bullish_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + bearish 1d trend + volume filter
            elif (close[i] < donchian_low[i] and 
                  ema_bearish_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend turns bearish
            if (close[i] < donchian_mid[i] or ema_bearish_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend turns bullish
            if (close[i] > donchian_mid[i] or ema_bullish_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals