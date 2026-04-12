#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Volume_TF
Hypothesis: Trade breakouts from daily Camarilla pivot levels with volume confirmation and 1-week ADX trend filter. 
Designed for 12-25 trades/year with strict entry conditions to avoid overtrading. 
Works in bull markets (breakouts continue) and bear markets (breakouts fail, reverse) by using trend filter to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Volume_TF"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    typical = (high + low + close) / 3
    range_ = high - low
    
    # Camarilla levels
    H5 = close + range_ * 1.1 / 2
    H4 = close + range_ * 1.1
    H3 = close + range_ * 1.1 * 0.5
    L3 = close - range_ * 1.1 * 0.5
    L4 = close - range_ * 1.1
    L5 = close - range_ * 1.1 / 2
    
    return H5, H4, H3, L3, L4, L5

def wilders_smooth(data, period):
    """Wilder's smoothing (same as RSI smoothing)."""
    result = np.full_like(data, np.nan)
    if len(data) < period:
        return result
    result[period-1] = np.nansum(data[:period])
    for i in range(period, len(data)):
        result[i] = result[i-1] - (result[i-1] / period) + data[i]
    return result

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(low)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA AND VOLUME ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each day
    H5_1d, H4_1d, H3_1d, L3_1d, L4_1d, L5_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate volume average (20-day)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === WEEKLY DATA FOR ADX TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on weekly
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, period=14)
    
    # Align all indicators to 12h timeframe
    H5_1d_aligned = align_htf_to_ltf(prices, df_1d, H5_1d)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    L5_1d_aligned = align_htf_to_ltf(prices, df_1d, L5_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(H5_1d_aligned[i]) or np.isnan(L5_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_1w_aligned[i] > 25
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma_1d_aligned[i] * 1.5)
        
        # Long: price breaks above H4 with volume and trend
        long_signal = (close[i] > H4_1d_aligned[i] and 
                      strong_volume and 
                      trending)
        
        # Short: price breaks below L4 with volume and trend
        short_signal = (close[i] < L4_1d_aligned[i] and 
                       strong_volume and 
                       trending)
        
        # Exit: price returns to H3/L3 or trend weakens
        exit_long = (position == 1 and 
                    (close[i] < H3_1d_aligned[i] or adx_1w_aligned[i] < 20))
        exit_short = (position == -1 and 
                     (close[i] > L3_1d_aligned[i] or adx_1w_aligned[i] < 20))
        
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