#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v2
Hypothesis: Refine breakout logic with stricter volume confirmation (3.0x avg) and add ADX filter (ADX > 20) to ensure trending conditions. Use discrete position sizing (0.0, ±0.30) to reduce fee churn. Target: 20-35 trades/year per symbol. Works in bull/bear via 1d trend filter and regime avoidance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla, trend, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34/ADX
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous 1d bar's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
    camarilla_s1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d ADX(14) for trend strength filter
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/window)
        atr = pd.Series(tr).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.maximum(atr, 1e-10)
        di_minus = 100 * dm_minus_smooth / np.maximum(atr, 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all HTF indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 3.0x 20-period average (stricter)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 3.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d EMA34/ADX) + volume MA
    start_idx = max(34, 20)  # 34 for EMA34/ADX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # 1d trend and strength filters
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        strong_trend = adx_1d_aligned[i] > 20  # ADX > 20 indicates trending market
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + strong trend + volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and trend_1d_uptrend and strong_trend and volume_spike[i]
            
            # Short: price breaks below S1 + 1d downtrend + strong trend + volume spike
            short_signal = (close[i] < camarilla_s1_aligned[i]) and trend_1d_downtrend and strong_trend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price touches S1 OR 1d trend turns down OR trend weakens (ADX < 20)
            if (close[i] < camarilla_s1_aligned[i] or not trend_1d_uptrend or not strong_trend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price touches R1 OR 1d trend turns up OR trend weakens
            if (close[i] > camarilla_r1_aligned[i] or not trend_1d_downtrend or not strong_trend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v2"
timeframe = "4h"
leverage = 1.0