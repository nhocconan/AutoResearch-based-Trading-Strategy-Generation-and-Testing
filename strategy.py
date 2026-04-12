#!/usr/bin/env python3
"""
6h_1d_WeeklyVWAP_Pullback
Hypothesis: Trade pullbacks to weekly VWAP on 6h chart with volume confirmation and 1d ADX trend filter. 
Weekly VWAP acts as dynamic support/resistance that institutions defend. Works in bull (buy pullbacks to VWAP) and bear (sell rallies to VWAP) markets.
Target: 60-100 total trades over 4 years (15-25/year) with clear trend-following logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyVWAP_Pullback"
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
    
    # === WEEKLY VWAP CALCULATION ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    volume_w = df_1w['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price_w = (high_w + low_w + close_w) / 3.0
    pv_w = typical_price_w * volume_w
    
    # Calculate cumulative sums for VWAP
    cum_pv = np.cumsum(pv_w)
    cum_vol = np.cumsum(volume_w)
    vwap_w = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Align weekly VWAP to 6h timeframe
    vwap_w_aligned = align_htf_to_ltf(prices, df_1w, vwap_w)
    
    # === DAILY ADX FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_d)
    minus_dm = np.zeros_like(low_d)
    tr = np.zeros_like(high_d)
    
    for i in range(1, len(high_d)):
        plus_dm[i] = max(high_d[i] - high_d[i-1], 0)
        minus_dm[i] = max(low_d[i-1] - low_d[i], 0)
        tr[i] = max(high_d[i] - low_d[i], 
                   abs(high_d[i] - close_d[i-1]), 
                   abs(low_d[i] - close_d[i-1]))
    
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
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6H INDICATORS ===
    # EMA20 for dynamic reference
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(vwap_w_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Distance from VWAP as percentage
        if vwap_w_aligned[i] != 0:
            dist_from_vwap = (close[i] - vwap_w_aligned[i]) / vwap_w_aligned[i]
        else:
            dist_from_vwap = 0
        
        # Long: pullback to VWAP support in uptrend
        long_signal = (dist_from_vwap <= 0.01 and  # Within 1% above VWAP
                      close[i] > vwap_w_aligned[i] and  # Above VWAP
                      ema20[i] > vwap_w_aligned[i] and  # EMA above VWAP (uptrend bias)
                      strong_volume and 
                      trending)
        
        # Short: rally to VWAP resistance in downtrend
        short_signal = (dist_from_vwap >= -0.01 and  # Within 1% below VWAP
                       close[i] < vwap_w_aligned[i] and  # Below VWAP
                       ema20[i] < vwap_w_aligned[i] and  # EMA below VWAP (downtrend bias)
                       strong_volume and 
                       trending)
        
        # Exit: price moves 2% away from VWAP or trend weakens
        exit_long = (position == 1 and 
                    (dist_from_vwap >= 0.02 or adx_aligned[i] < 15))
        exit_short = (position == -1 and 
                     (dist_from_vwap <= -0.02 or adx_aligned[i] < 15))
        
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