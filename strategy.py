#!/usr/bin/env python3
"""
12h_1d_Camarilla_Trend_Follow_v1
Hypothesis: Trade Camarilla pivot breakouts on 12h timeframe with 1-day trend filter.
Long when price breaks above H3 with 1-day ADX > 25, short when breaks below L3 with 1-day ADX > 25.
Use volume confirmation (>1.5x 20-period average) to avoid false breakouts.
Designed for 15-25 trades/year per symbol with strong trend-following logic that works in bull (breakouts continue) and bear (breakouts fail, reverse) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Trend_Follow_v1"
timeframe = "12h"
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
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
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
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12H INDICATORS: CAMARILLA PIVOT LEVELS ===
    # Calculate pivot points from previous day
    pivots_high = np.full(n, np.nan)
    pivots_low = np.full(n, np.nan)
    pivots_close = np.full(n, np.nan)
    
    # Map 12h bars to previous day's OHLC
    for i in range(n):
        # Get the date of current 12h bar
        current_date = pd.Timestamp(prices.iloc[i]['open_time']).date()
        # Previous trading day
        prev_date = current_date - pd.Timedelta(days=1)
        
        # Find index of previous day in 1d data
        prev_day_idx = None
        for j in range(len(df_1d)):
            if pd.Timestamp(df_1d.iloc[j]['open_time']).date() == prev_date:
                prev_day_idx = j
                break
        
        if prev_day_idx is not None:
            ph = high_1d[prev_day_idx]
            pl = low_1d[prev_day_idx]
            pc = close_1d[prev_day_idx]
            
            pivots_high[i] = ph
            pivots_low[i] = pl
            pivots_close[i] = pc
    
    # Calculate Camarilla levels
    H3 = pivots_close + (pivots_high - pivots_low) * 1.1 / 4
    L3 = pivots_close - (pivots_high - pivots_low) * 1.1 / 4
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Trend filter: ADX > 25 indicates strong trending market
        trending = adx_aligned[i] > 25
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price breaks above H3 with volume and trend
        long_signal = (close[i] > H3[i] and 
                      strong_volume and 
                      trending)
        
        # Short: price breaks below L3 with volume and trend
        short_signal = (close[i] < L3[i] and 
                       strong_volume and 
                       trending)
        
        # Exit: price returns to pivot level or trend weakens
        exit_long = (position == 1 and 
                    (close[i] < pivots_close[i] or adx_aligned[i] < 20))
        exit_short = (position == -1 and 
                     (close[i] > pivots_close[i] or adx_aligned[i] < 20))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.30
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals