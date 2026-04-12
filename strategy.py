#!/usr/bin/env python3
"""
4h_1d_SuperTrend_Pullback_v1
Hypothesis: Enter on pullbacks to SuperTrend during strong trends (1d ADX > 25) with volume confirmation. 
SuperTrend (ATR=10, multiplier=3) provides dynamic support/resistance. Works in bull (buy dips in uptrend) 
and bear (sell rallies in downtrend) by following the trend direction. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_SuperTrend_Pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR ADX TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4H INDICATORS: SUPERTREND ===
    atr_period = 10
    multiplier = 3
    
    # Calculate True Range
    tr_4h = np.zeros_like(high)
    for i in range(1, len(high)):
        tr_4h[i] = max(high[i] - low[i], 
                      abs(high[i] - close[i-1]), 
                      abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr_4h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    
    for i in range(len(close)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # SuperTrend
    supertrend = np.zeros_like(close)
    trend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = final_ub[i]
            trend[i] = 1
        else:
            if trend[i-1] == 1 and close[i] <= final_ub[i]:
                trend[i] = -1
                supertrend[i] = final_lb[i]
            elif trend[i-1] == -1 and close[i] >= final_lb[i]:
                trend[i] = 1
                supertrend[i] = final_ub[i]
            else:
                trend[i] = trend[i-1]
                supertrend[i] = final_ub[i] if trend[i] == 1 else final_lb[i]
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(supertrend[i]) or np.isnan(trend[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: pullback to SuperTrend in uptrend
        long_signal = (trend[i] == 1 and 
                      close[i] <= supertrend[i] * 1.005 and  # Allow small overshoot
                      strong_trend and 
                      strong_volume)
        
        # Short: rally to SuperTrend in downtrend
        short_signal = (trend[i] == -1 and 
                       close[i] >= supertrend[i] * 0.995 and  # Allow small overshoot
                       strong_trend and 
                       strong_volume)
        
        # Exit: trend changes or price moves significantly away from SuperTrend
        exit_long = (position == 1 and 
                    (trend[i] == -1 or close[i] > supertrend[i] * 1.02))
        exit_short = (position == -1 and 
                     (trend[i] == 1 or close[i] < supertrend[i] * 0.98))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals