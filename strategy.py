#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.5x average AND 12h EMA34 > EMA89 (uptrend).
Short when price breaks below Camarilla S1 AND volume > 1.5x average AND 12h EMA34 < EMA89 (downtrend).
Exit when price reverts to Camarilla midpoint (PP) OR trend reverses.
Uses 4h for price/volume, 12h for EMA trend filter to avoid whipsaw.
Target: 50-150 total trades over 4 years (12-38/year). Camarilla levels provide precise intraday pivot points,
volume confirmation reduces fakeouts, 12h EMA crossover ensures we trade with the intermediate trend.
Works in bull markets (captures uptrend breakouts) and bear markets (captures downtrend breakdowns).
"""

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
    
    # Get 4h data for Camarilla calculation and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate previous day's Camarilla levels (using prior 4h day's OHLC)
    # Need to group 4h bars into days - 6 bars per 4h day
    # We'll use rolling window of 6 periods to get daily OHLC
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    close_series = pd.Series(close_4h)
    
    # Daily OHLC from 4h data (6 bars = 1 day)
    daily_high = high_series.rolling(window=6, min_periods=6).max().shift(6)  # prior day
    daily_low = low_series.rolling(window=6, min_periods=6).min().shift(6)   # prior day
    daily_close = close_series.rolling(window=6, min_periods=6).last().shift(6) # prior day
    daily_open = close_series.rolling(window=6, min_periods=6).first().shift(6) # prior day open
    
    # Typical price for pivot calculation
    typical_price = (daily_high + daily_low + daily_close) / 3
    
    # Camarilla levels
    R1 = typical_price + 1.1 * (daily_high - daily_low) / 12
    S1 = typical_price - 1.1 * (daily_high - daily_low) / 12
    PP = typical_price  # pivot point
    
    # Calculate volume average (20-period) on 4h
    volume_series = pd.Series(volume_4h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34 = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = close_12h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align all indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    PP_aligned = align_htf_to_ltf(prices, df_4h, PP)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    ema_89_aligned = align_htf_to_ltf(prices, df_12h, ema_89)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        pp = PP_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema34 = ema_34_aligned[i]
        ema89 = ema_89_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.5x avg AND 12h EMA34 > EMA89 (uptrend)
            if price > r1 and vol > 1.5 * vol_ma and ema34 > ema89:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.5x avg AND 12h EMA34 < EMA89 (downtrend)
            elif price < s1 and vol > 1.5 * vol_ma and ema34 < ema89:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < PP OR trend reverses (EMA34 < EMA89)
            if price < pp or ema34 < ema89:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > PP OR trend reverses (EMA34 > EMA89)
            if price > pp or ema34 > ema89:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA_Trend_Filter_v2"
timeframe = "4h"
leverage = 1.0