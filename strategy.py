# 6h_1d_ewz_volume_trend_v1
# Hypothesis: 6h Ehlers Zero Lag (EZL) with 1d trend filter and volume confirmation
# EZL reduces lag while maintaining smoothness - effective in both trending and ranging markets
# 1d trend filter ensures alignment with higher timeframe direction
# Volume confirmation filters out false breakouts
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)
# Works in bull markets via trend following, in bear via mean reversion at extremes

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ewz_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ehlers Zero Lag (EZL) indicator
    # EZL = 2*EMA(price, n) - EMA(EMA(price, n), n) where n = sqrt(period)
    period = 20
    lag = int(np.sqrt(period))
    ema1 = pd.Series(close).ewm(span=lag, adjust=False, min_periods=lag).mean().values
    ema2 = pd.Series(ema1).ewm(span=lag, adjust=False, min_periods=lag).mean().values
    ezl = 2 * ema1 - ema2
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ezl[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # EZL signals: price crossing above/below EZL with slope confirmation
        ezl_slope = ezl[i] - ezl[i-1] if i > 0 else 0
        ezl_cross_up = close[i] > ezl[i] and close[i-1] <= ezl[i-1]
        ezl_cross_down = close[i] < ezl[i] and close[i-1] >= ezl[i-1]
        
        long_entry = ezl_cross_up and volume_filter and is_uptrend and ezl_slope > 0
        short_entry = ezl_cross_down and volume_filter and is_downtrend and ezl_slope < 0
        
        # Exit when price crosses back through EZL or trend changes
        long_exit = ezl_cross_down or (not is_uptrend)
        short_exit = ezl_cross_up or (not is_downtrend)
        
        # Priority: entry > exit > hold
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals