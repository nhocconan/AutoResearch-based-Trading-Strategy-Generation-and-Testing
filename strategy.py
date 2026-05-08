#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Uses daily EMA34 for trend direction, 1d Camarilla levels for entry/exit signals,
# and volume breakout (>1.5x average) for confirmation. Designed to work in both bull and bear
# markets by following daily trend while using Camarilla levels as dynamic support/resistance.
# Target: 12-30 trades/year (50-120 total over 4 years).

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for trend filter and Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_r4 = np.full(len(close_daily), np.nan)
    camarilla_r3 = np.full(len(close_daily), np.nan)
    camarilla_s3 = np.full(len(close_daily), np.nan)
    camarilla_s4 = np.full(len(close_daily), np.nan)
    
    for i in range(1, len(close_daily)):
        # Use previous day's OHLC to calculate today's levels
        prev_high = high_daily[i-1]
        prev_low = low_daily[i-1]
        prev_close = close_daily[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_r4[i] = prev_close + range_val * 1.5
            camarilla_r3[i] = prev_close + range_val * 1.25
            camarilla_s3[i] = prev_close - range_val * 1.25
            camarilla_s4[i] = prev_close - range_val * 1.5
        else:
            camarilla_r4[i] = camarilla_r3[i] = camarilla_s3[i] = camarilla_s4[i] = prev_close
    
    # Calculate daily volume average for volume breakout
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 6h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 1)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current 6h volume > 1.5x 20-period average of daily volume
        vol_breakout = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_breakout = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R3/S3 in direction of daily trend
            # Bullish breakout: price breaks above R3 with daily uptrend
            bullish_breakout = (
                close[i] > camarilla_r3_aligned[i] and   # price above R3
                close[i-1] <= camarilla_r3_aligned[i-1] and  # was below or at R3 (breakout)
                ema34_daily_aligned[i] > close[i] * 0.995 and  # daily EMA above price (uptrend filter)
                vol_breakout
            )
            
            # Bearish breakout: price breaks below S3 with daily downtrend
            bearish_breakout = (
                close[i] < camarilla_s3_aligned[i] and   # price below S3
                close[i-1] >= camarilla_s3_aligned[i-1] and  # was above or at S3 (breakdown)
                ema34_daily_aligned[i] < close[i] * 1.005 and  # daily EMA below price (downtrend filter)
                vol_breakout
            )
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or reaches R4 (take profit)
            if close[i] < camarilla_r3_aligned[i] or close[i] > camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or reaches S4 (take profit)
            if close[i] > camarilla_s3_aligned[i] or close[i] < camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals