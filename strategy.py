#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation
# Long when: price breaks above Donchian(20) upper band AND 12h EMA50 > EMA200 AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND 12h EMA50 < EMA200 AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian(20) middle band OR volume drops below average
# Uses Donchian for structure, 12h EMA for trend filter, volume for conviction
# Timeframe: 4h, HTF: 12h. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_DonchianBreakout_12hEMATrend_VolumeConfirm"
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
    
    # Get 12h data ONCE before loop for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA200
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 200:
        ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    else:
        ema50_12h = np.full(len(close_12h), np.nan)
        ema200_12h = np.full(len(close_12h), np.nan)
    
    # 12h EMA trend: bullish when EMA50 > EMA200, bearish when EMA50 < EMA200
    ema_bullish = ema50_12h > ema200_12h
    ema_bearish = ema50_12h < ema200_12h
    
    # Align 12h EMA trend to 4h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_bullish.astype(float))
    ema_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_bearish.astype(float))
    
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
            # Long conditions: break above Donchian high + bullish 12h EMA trend + volume filter
            if (close[i] > donchian_high[i] and 
                ema_bullish_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + bearish 12h EMA trend + volume filter
            elif (close[i] < donchian_low[i] and 
                  ema_bearish_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR volume drops below average
            if (close[i] < donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR volume drops below average
            if (close[i] > donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals