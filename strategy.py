#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation.
- Uses 6h timeframe (primary) and 1d HTF for ADX trend alignment
- Ichimoku components calculated on 6h: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displacement)
- Breakout logic: long when price crosses above Kumo (cloud) with bullish TK cross and volume confirmation in uptrend (ADX>25)
  short when price crosses below Kumo with bearish TK cross and volume confirmation in downtrend (ADX>25)
- Trend filter: only trade when 1d ADX > 25 (strong trend) to avoid ranging markets
- Volume confirmation: current volume > 1.3 * 20-period volume MA to filter low-quality breakouts
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe as per research
- Works in both bull/bear: ADX filter ensures we only trade strong trends, Ichimoku provides dynamic support/resistance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter (strength only, not direction)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Rest: Wilder's smoothing
        alpha = 1 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = result[i-1]
        return result
    
    atr = wilders_smooth(tr, 30)
    plus_di = 100 * wilders_smooth(plus_dm, 30) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 30) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 30)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 25
    
    # Calculate Ichimoku on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Current Kumo (cloud) bounds: Senkou Span A/B from 26 periods ago
    # Since Senkou spans are plotted ahead, the current cloud uses values from 26 periods back
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # TK Cross
    tk_cross_bullish = tenkan > kijun
    tk_cross_bearish = tenkan < kijun
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52 + 26, 30, 20)  # Senkou B lookback + displacement, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud AND bullish TK cross AND strong trend AND volume confirmation
            if (close[i] > cloud_top[i] and tk_cross_bullish[i] and strong_trend[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND bearish TK cross AND strong trend AND volume confirmation
            elif (close[i] < cloud_bottom[i] and tk_cross_bearish[i] and strong_trend[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud base OR bearish TK cross
            if close[i] < cloud_bottom[i] or not tk_cross_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud top OR bullish TK cross
            if close[i] > cloud_top[i] or not tk_cross_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0