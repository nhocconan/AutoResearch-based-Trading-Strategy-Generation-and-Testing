#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses discrete position sizing (0.20) to minimize fee churn. Combines mean-reversion pivot breaks with
# higher-timeframe trend filtering for robustness in both bull and bear markets. Session filter (08-20 UTC) reduces noise.
# Target: 15-37 trades/year per symbol (60-150 over 4 years) for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Trend_Session"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - avoids datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: based on previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d_prev + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d_prev - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe (using previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1h data for volume EMA(20) for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Calculate 1h volume EMA(20) for volume confirmation
    vol_1h = df_1h['volume'].values
    vol_ema_20 = pd.Series(vol_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        # 4h trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_4h_aligned[i]
        bearish_trend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + volume confirmation + bullish 4h trend
            if (close[i] > camarilla_r3_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + volume confirmation + bearish 4h trend
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below Camarilla S3 OR 4h trend turns bearish
            if close[i] < camarilla_s3_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above Camarilla R3 OR 4h trend turns bullish
            if close[i] > camarilla_r3_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals