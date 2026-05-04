#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot points identify key intraday support/resistance levels.
# Breakout above R3 or below S3 with volume confirmation signals strong momentum.
# 4h EMA50 provides higher-timeframe trend bias to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise during low-liquidity periods.
# Designed for 1h timeframe targeting 60-150 total trades over 4 years (15-37/year).
# Works in both bull and bear markets via trend-filtered breakouts.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
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
    
    # Pre-compute session filter (08-20 UTC)
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
    
    # Get daily data for Camarilla pivot points (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today (based on yesterday's OHLC)
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align daily Camarilla levels to 1h timeframe (yesterday's levels for today's trading)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume EMA(20) on 1h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        # 4h trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_4h_aligned[i]
        bearish_trend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: break above R3 + volume confirmation + bullish 4h trend
            if (close[i] > camarilla_r3_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.20
                position = 1
            # Short: break below S3 + volume confirmation + bearish 4h trend
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla H-L range (mean reversion) OR 4h trend turns bearish
            camarilla_h3 = camarilla_r3_aligned[i]  # R3 is approx H3 for exit
            camarilla_l3 = camarilla_s3_aligned[i]  # S3 is approx L3 for exit
            if (close[i] < camarilla_h3 and close[i] > camarilla_l3) or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters Camarilla H-L range OR 4h trend turns bullish
            camarilla_h3 = camarilla_r3_aligned[i]
            camarilla_l3 = camarilla_s3_aligned[i]
            if (close[i] < camarilla_h3 and close[i] > camarilla_l3) or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals