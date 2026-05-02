#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses weekly pivot direction from 1w timeframe to filter breakout direction
# Camarilla levels from 1d provide institutional reference points for mean reversion/breakout
# Breakout at R3/S3 with volume spike confirms institutional participation
# 1d EMA34 trend filter ensures alignment with daily trend
# Weekly pivot direction adds higher timeframe bias to reduce counter-trend trades
# Target: 75-150 total trades over 4 years (19-37/year) to balance opportunity and fee drag
# Discrete position sizing: 0.25 (25% of capital) to minimize fee churn while maintaining reasonable exposure

name = "6h_Camarilla_R3_S3_Breakout_1wPivotDir_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla calculation: based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivots and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + range * 1.1/4, S3 = close - range * 1.1/4
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Calculate 1w EMA50 for trend filter (weekly pivot direction proxy)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA34 for trend filter
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Determine weekly trend direction from 1w EMA50
            weekly_bullish = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False
            weekly_bearish = close_1w[-1] < ema_50_1w[-1] if len(close_1w) > 0 else False
            
            # Long entry: price breaks above R3 with volume spike AND price > 1d EMA34 (bullish trend) AND weekly bullish bias
            if (close[i] > r3_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i] and
                weekly_bullish):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 1d EMA34 (bearish trend) AND weekly bearish bias
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  weekly_bearish):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 OR below 1d EMA34 (trend change) OR weekly turn bearish
            if (close[i] < s3_1d_aligned[i] or 
                close[i] < ema_34_1d_aligned[i] or
                not weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 OR above 1d EMA34 (trend change) OR weekly turn bullish
            if (close[i] > r3_1d_aligned[i] or 
                close[i] > ema_34_1d_aligned[i] or
                weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals