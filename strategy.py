#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w trend filter and volume confirmation
# Long when price breaks above Camarilla R4 resistance AND 1w bullish trend (close > EMA50) AND volume > 2x 20-period volume EMA
# Short when price breaks below Camarilla S4 support AND 1w bearish trend (close < EMA50) AND volume > 2x 20-period volume EMA
# Uses Camarilla R4/S4 (extreme levels) for institutional breakouts; 1w EMA50 for major trend filter; high volume threshold to reduce false signals.
# Designed for 1d timeframe: targets 7-25 trades/year (30-100 total over 4 years) with discrete position sizing (0.30) to minimize fee drag.
# Works in bull markets via longs in bullish 1w trend and bear markets via shorts in bearish 1w trend.

name = "1d_Camarilla_R4S4_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Get prior week's OHLC for Camarilla levels (use 1w data - same df_1w)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R4 and S4 calculation:
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    camarilla_r4_1w = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s4_1w = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align prior week's Camarilla levels to 1d timeframe (wait for week to complete)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND 1w bullish trend AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND 1w bearish trend AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S4 OR 1w trend turns bearish
            if (close[i] < camarilla_s4_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Camarilla R4 OR 1w trend turns bullish
            if (close[i] > camarilla_r4_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals