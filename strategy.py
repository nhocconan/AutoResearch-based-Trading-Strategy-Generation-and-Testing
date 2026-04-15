#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1h Trend Filter and Volume Spike
# Williams %R(14) identifies overbought/oversold conditions. 
# Long when %R crosses above -80 from below (oversold bounce) in uptrend (1h EMA20 > EMA50).
# Short when %R crosses below -20 from above (overbought rejection) in downtrend (1h EMA20 < EMA50).
# Volume spike > 2x median confirms momentum. Works in bull (buy dips) and bear (sell rallies).
# Target: 50-150 total trades with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 1h EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    close_1h = df_1h['close'].values
    ema20 = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: 1 = uptrend (EMA20 > EMA50), -1 = downtrend (EMA20 < EMA50), 0 = neutral
    trend = np.where(ema20 > ema50, 1, np.where(ema20 < ema50, -1, 0))
    trend_aligned = align_htf_to_ltf(prices, df_1h, trend)
    
    # Williams %R cross signals
    willr_long_signal = (willr > -80) & (np.roll(willr, 1) <= -80)
    willr_short_signal = (willr < -20) & (np.roll(willr, 1) >= -20)
    
    # Volume spike > 2x median of last 20 periods
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    vol_spike = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if trend data not available
        if np.isnan(trend_aligned[i]):
            continue
            
        # Long: Williams %R oversold bounce + uptrend + volume spike
        if (willr_long_signal[i] and 
            trend_aligned[i] == 1 and 
            vol_spike[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: Williams %R overbought rejection + downtrend + volume spike
        elif (willr_short_signal[i] and 
              trend_aligned[i] == -1 and 
              vol_spike[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Williams %R signal or trend change
        elif position == 1 and (willr_short_signal[i] or trend_aligned[i] == -1):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (willr_long_signal[i] or trend_aligned[i] == 1):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_Trend_Volume"
timeframe = "4h"
leverage = 1.0