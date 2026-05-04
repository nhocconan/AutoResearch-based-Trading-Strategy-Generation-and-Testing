#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 resistance AND 12h bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when price breaks below Camarilla S1 support AND 12h bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Uses Camarilla R1/S1 (stronger levels) for fewer, higher-quality breaks; 12h EMA50 for smoother trend filter; volume confirmation to reduce false breakouts.
# Designed for 4h timeframe: targets 19-50 trades/year (75-200 total over 4 years) with discrete position sizing (0.30) to minimize fee drag.
# Works in bull markets via longs in bullish 12h trend and bear markets via shorts in bearish 12h trend.

name = "4h_Camarilla_R1S1_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_12h = close_12h > ema_50_12h
    trend_bearish_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish_12h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish_12h.astype(float))
    
    # Get prior 12h's OHLC for Camarilla levels (use 12h data - same df_12h)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R1 and S1 calculation:
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12
    camarilla_s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Align prior 12h's Camarilla levels to 4h timeframe (wait for 12h to complete)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND 12h bullish trend AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 12h bullish trend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND 12h bearish trend AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 12h bearish trend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S1 OR 12h trend turns bearish
            if (close[i] < camarilla_s1_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Camarilla R1 OR 12h trend turns bullish
            if (close[i] > camarilla_r1_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals