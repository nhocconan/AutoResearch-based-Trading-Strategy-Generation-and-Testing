#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and 1w trend filter.
    # Donchian breakouts capture momentum; ATR filter ensures high volatility regime.
    # 1w EMA200 filter avoids counter-trend trades in strong trends.
    # Works in bull/bear via volatility regime targeting and trend alignment.
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR volatility filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 12h data for Donchian channels (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA200 for trend filter
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: current 1d ATR > 50-period mean (high volatility regime)
        atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
        volatility_filter = atr_aligned[i] > atr_ma_aligned[i]
        
        # Trend filter: price above/below 1w EMA200
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > upper_20_aligned[i]  # Break above upper band
        breakout_short = close[i] < lower_20_aligned[i]  # Break below lower band
        
        # Entry conditions: breakout with volatility filter and trend alignment
        long_entry = breakout_long and volatility_filter and uptrend
        short_entry = breakout_short and volatility_filter and downtrend
        
        # Exit conditions: price returns to opposite Donchian band
        long_exit = close[i] < lower_20_aligned[i]
        short_exit = close[i] > upper_20_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_donchian_atr_trend_v1"
timeframe = "12h"
leverage = 1.0