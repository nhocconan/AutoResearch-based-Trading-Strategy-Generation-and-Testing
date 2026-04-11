#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v3
# Strategy: 4h Camarilla breakout with 1d volume confirmation and ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Breakouts above Camarilla H4 (resistance) or below L4 (support) levels
# calculated from prior 1-day OHLC, with above-average daily volume and strong trend (ADX > 25).
# Works in bull markets (breakout longs) and bear markets (breakdown shorts) by capturing
# momentum after consolidation. Uses Camarilla levels for precise entry points and volume/ADX
# for confirmation to reduce false signals. Target: 20-40 trades/year to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v3"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    if len(tr) < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

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
    
    # Calculate Camarilla levels from prior 1-day OHLC
    # Typical Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # Where C = close, H = high, L = low of prior day
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        vol_confirm = vol_1d_current > 1.5 * vol_avg_20_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > camarilla_h4_aligned[i-1]  # break above prior H4
        breakout_down = close[i] < camarilla_l4_aligned[i-1]  # break below prior L4
        
        long_signal = breakout_up and vol_confirm and trend_filter
        short_signal = breakout_down and vol_confirm and trend_filter
        
        # Exit conditions: opposite breakout or trend weakening
        long_exit = close[i] < camarilla_l4_aligned[i-1] or adx_aligned[i] < 20
        short_exit = close[i] > camarilla_h4_aligned[i-1] or adx_aligned[i] < 20
        
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