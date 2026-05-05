#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above Ichimoku cloud (Senkou Span A) AND 1d close > 1d EMA50 AND volume > 1.5 * avg volume
# Short when price breaks below Ichimoku cloud (Senkou Span B) AND 1d close < 1d EMA50 AND volume > 1.5 * avg volume
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Ichimoku cloud acts as dynamic support/resistance; EMA50 filters trend; volume confirms breakout strength.
# Works in bull markets via longs above cloud and bear markets via shorts below cloud.

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need 26*2 for Ichimoku
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Shift Senkou Spans forward by 26 periods (cloud is plotted ahead)
    senkou_span_a = np.roll(senkou_span_a, 26)
    senkou_span_b = np.roll(senkou_span_b, 26)
    senkou_span_a[:26] = np.nan
    senkou_span_b[:26] = np.nan
    
    # Align Ichimoku cloud to 6h timeframe (actually already 6h, but for consistency)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 6h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Calculate average volume for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # ~6 days of 6h bars
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Senkou Span A (top of cloud) AND 1d uptrend AND high volume
            if (close[i] > senkou_span_a_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Senkou Span B (bottom of cloud) AND 1d downtrend AND high volume
            elif (close[i] < senkou_span_b_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Senkou Span B (bottom of cloud) OR 1d trend changes to downtrend
            if (close[i] < senkou_span_b_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Senkou Span A (top of cloud) OR 1d trend changes to uptrend
            if (close[i] > senkou_span_a_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals