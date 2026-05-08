#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Trend Filter + Volume Confirmation
# Uses 1d EMA50 for trend bias, 6h Ichimoku TK cross for entry timing, and volume > 1.5x average for confirmation.
# Designed to work in both bull and bear markets by following daily trend while avoiding false signals.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Ichimoku_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Get 6h data for Ichimoku components
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need 26*2 for Ichimoku
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full(len(high_6h), np.nan)
    period9_low = np.full(len(low_6h), np.nan)
    if len(high_6h) >= 9:
        for i in range(9, len(high_6h)):
            period9_high[i] = np.max(high_6h[i-9:i+1])
            period9_low[i] = np.min(low_6h[i-9:i+1])
    tenkan_sen = np.full(len(high_6h), np.nan)
    if len(high_6h) >= 9:
        for i in range(9, len(high_6h)):
            tenkan_sen[i] = (period9_high[i] + period9_low[i]) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full(len(high_6h), np.nan)
    period26_low = np.full(len(low_6h), np.nan)
    if len(high_6h) >= 26:
        for i in range(26, len(high_6h)):
            period26_high[i] = np.max(high_6h[i-26:i+1])
            period26_low[i] = np.min(low_6h[i-26:i+1])
    kijun_sen = np.full(len(high_6h), np.nan)
    if len(high_6h) >= 26:
        for i in range(26, len(high_6h)):
            kijun_sen[i] = (period26_high[i] + period26_low[i]) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full(len(high_6h), np.nan)
    if len(high_6h) >= 26:
        for i in range(len(high_6h)):
            idx = i + 26
            if idx < len(tenkan_sen) and not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.full(len(high_6h), np.nan)
    period52_low = np.full(len(low_6h), np.nan)
    if len(high_6h) >= 52:
        for i in range(52, len(high_6h)):
            period52_high[i] = np.max(high_6h[i-52:i+1])
            period52_low[i] = np.min(low_6h[i-52:i+1])
    senkou_span_b = np.full(len(high_6h), np.nan)
    if len(high_6h) >= 52:
        for i in range(len(high_6h)):
            idx = i + 26
            if idx < len(period52_high) and not np.isnan(period52_high[i]) and not np.isnan(period52_low[i]):
                senkou_span_b[idx] = (period52_high[i] + period52_low[i]) / 2
    
    # Calculate daily volume average for volume confirmation
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 6h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Align Ichimoku components to 6h timeframe (they're already on 6h, but need alignment for safety)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 26, 52, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_daily_aligned[i]) or np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average of daily volume
        vol_confirm = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Look for entry: TK cross in direction of daily trend with volume confirmation
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
            
            # Long when bullish TK cross, price above cloud, and daily trend is up
            long_condition = (
                tk_cross_bullish and
                close[i] > cloud_top and
                close[i] > ema50_daily_aligned[i] and  # price above daily EMA50 (bullish bias)
                vol_confirm
            )
            
            # Short when bearish TK cross, price below cloud, and daily trend is down
            short_condition = (
                tk_cross_bearish and
                close[i] < cloud_bottom and
                close[i] < ema50_daily_aligned[i] and  # price below daily EMA50 (bearish bias)
                vol_confirm
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Kijun or enters cloud
            if tenkan_sen_aligned[i] < kijun_sen_aligned[i] or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Kijun or enters cloud
            if tenkan_sen_aligned[i] > kijun_sen_aligned[i] or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals