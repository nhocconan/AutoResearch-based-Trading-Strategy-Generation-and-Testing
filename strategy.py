#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# Long when price breaks above Camarilla R3 resistance AND 4h bullish trend (close > EMA20) AND volume > 1.5x 20-period volume EMA
# Short when price breaks below Camarilla S3 support AND 4h bearish trend (close < EMA20) AND volume > 1.5x 20-period volume EMA
# Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods
# Uses 1h timeframe for entry timing, 4h for signal direction to reduce overtrading
# Target: 15-37 trades/year (60-150 total over 4 years) with discrete position sizing (0.20) to minimize fee drag
# Works in bull markets via longs in bullish 4h trend and bear markets via shorts in bearish 4h trend

name = "1h_Camarilla_R3S3_4hTrend_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_bullish_4h = close_4h > ema_20_4h
    trend_bearish_4h = close_4h < ema_20_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish_4h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish_4h.astype(float))
    
    # Get prior day's OHLC for Camarilla levels (use 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 calculation:
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align prior day's Camarilla levels to 1h timeframe (wait for day to complete)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 4h bullish trend AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 4h bullish trend
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 4h bearish trend AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 4h bearish trend
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 4h trend turns bearish
            if (close[i] < camarilla_s3_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 4h trend turns bullish
            if (close[i] > camarilla_r3_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals