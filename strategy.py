#!/usr/bin/env python3
# 4h_1d_camarilla_pullback_v3
# Strategy: 4h Camarilla pivot pullback with 1d volume confirmation and ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price often pulls back to Camarilla pivot levels (H3/L3) before continuing the trend.
# In strong trends (ADX > 25), buying near L3 in uptrends or selling near H3 in downtrends
# with above-average 1d volume offers high-probability entries. Works in bull markets
# (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends).
# Target: 20-40 trades/year to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pullback_v3"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = close + (range_val * 1.1 / 12)
    d = close - (range_val * 1.1 / 12)
    h3 = close + (range_val * 1.1 / 6)
    l3 = close - (range_val * 1.1 / 6)
    h4 = close + (range_val * 1.1 / 2)
    l4 = close - (range_val * 1.1 / 2)
    return h3, l3, h4, l4

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
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr14 = wilders_smooth(tr, period)
    dm_plus_14 = wilders_smooth(dm_plus, period)
    dm_minus_14 = wilders_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d volume average (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate daily Camarilla levels
    h3_1d = np.zeros(len(df_1d))
    l3_1d = np.zeros(len(df_1d))
    h4_1d = np.zeros(len(df_1d))
    l4_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        h3, l3, h4, l4 = calculate_camarilla(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i]
        )
        h3_1d[i] = h3
        l3_1d[i] = l3
        h4_1d[i] = h4
        l4_1d[i] = l4
    
    # Align Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        vol_confirm = vol_1d_current > 1.3 * vol_avg_20_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        # Pullback to Camarilla levels
        # Long: pullback to L3 in uptrend (price near L3 and rising)
        # Short: pullback to H3 in downtrend (price near H3 and falling)
        near_l3 = abs(close[i] - l3_aligned[i]) < (h4_aligned[i] - l4_aligned[i]) * 0.02
        near_h3 = abs(close[i] - h3_aligned[i]) < (h4_aligned[i] - l4_aligned[i]) * 0.02
        
        # Price momentum confirmation
        price_up = close[i] > close[i-3]  # rising over 3 periods
        price_down = close[i] < close[i-3]  # falling over 3 periods
        
        long_signal = near_l3 and price_up and vol_confirm and trend_filter
        short_signal = near_h3 and price_down and vol_confirm and trend_filter
        
        # Exit conditions: opposite Camarilla level or trend weakening
        long_exit = (close[i] > h3_aligned[i]) or (adx_aligned[i] < 20)
        short_exit = (close[i] < l3_aligned[i]) or (adx_aligned[i] < 20)
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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