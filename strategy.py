#!/usr/bin/env python3
"""
4h_Ichimoku_Kumo_Twist_RSI_Filter
Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) signals trend changes, RSI(14) confirms momentum strength, and 1d ADX filters range conditions. Long when price > Kumo, Senkou A > Senkou B (bullish twist), RSI > 50, and 1d ADX > 25. Short when price < Kumo, Senkou A < Senkou B (bearish twist), RSI < 50, and 1d ADX > 25. Uses volume confirmation (current volume > 20-period average) to avoid false breaks. Target: 20-40 trades/year per symbol.
"""

name = "4h_Ichimoku_Kumo_Twist_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Kumo (Cloud): Senkou Span A and B shifted 26 periods ahead
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    # Set first 26 values to NaN (will be handled by NaN checks)
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku calculations are valid
        # Skip if any critical values are NaN
        if np.isnan(senkou_span_a_shifted[i]) or np.isnan(senkou_span_b_shifted[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Kumo twist conditions
        senkou_a = senkou_span_a_shifted[i]
        senkou_b = senkou_span_b_shifted[i]
        bullish_twist = senkou_a > senkou_b
        bearish_twist = senkou_a < senkou_b
        
        # Price vs Kumo
        price_above_kumo = close[i] > max(senkou_a, senkou_b)
        price_below_kumo = close[i] < min(senkou_a, senkou_b)
        
        # RSI condition
        rsi_ok_long = rsi[i] > 50
        rsi_ok_short = rsi[i] < 50
        
        # ADX trend strength filter (trending market only)
        adx_ok = adx_aligned[i] > 25
        
        # Volume confirmation
        vol_ok = volume_ok[i]
        
        if position == 0:
            # LONG: price > Kumo, bullish twist, RSI > 50, ADX > 25, volume OK
            if price_above_kumo and bullish_twist and rsi_ok_long and adx_ok and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Kumo, bearish twist, RSI < 50, ADX > 25, volume OK
            elif price_below_kumo and bearish_twist and rsi_ok_short and adx_ok and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Kumo or bearish twist or RSI < 50
            if price_below_kumo or bearish_twist or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Kumo or bullish twist or RSI > 50
            if price_above_kumo or bullish_twist or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals