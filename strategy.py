#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when: price breaks above 20-period Donchian high AND 1d EMA34 > EMA89 (bullish trend) AND volume > 1.5x 20-period MA
# Short when: price breaks below 20-period Donchian low AND 1d EMA34 < EMA89 (bearish trend) AND volume > 1.5x 20-period MA
# Exit when: price returns to opposite Donchian level OR volume drops below average
# Uses Donchian for breakout structure, 1d EMA for trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian_1dEMA_VolumeConfirm"
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
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # need sufficient data for EMA89
        return np.zeros(n)
    
    # Calculate EMA34 and EMA89 on 1d
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89 = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # EMA trend signals
    ema_bullish = np.zeros(len(ema34), dtype=bool)
    ema_bearish = np.zeros(len(ema34), dtype=bool)
    for i in range(len(ema34)):
        if not np.isnan(ema34[i]) and not np.isnan(ema89[i]):
            ema_bullish[i] = ema34[i] > ema89[i]
            ema_bearish[i] = ema34[i] < ema89[i]
    
    # Align 1d EMA trends to 4h timeframe
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
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high + bullish 1d EMA + volume filter
            if (close[i] > donch_high[i] and 
                ema_bullish_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + bearish 1d EMA + volume filter
            elif (close[i] < donch_low[i] and 
                  ema_bearish_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low OR volume filter fails
            if (close[i] < donch_low[i] or volume_filter[i] == 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high OR volume filter fails
            if (close[i] > donch_high[i] or volume_filter[i] == 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals