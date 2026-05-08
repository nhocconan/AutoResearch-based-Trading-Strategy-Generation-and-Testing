#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 1d Trend Filter and Volume Spike
# Uses daily EMA34 for trend direction, Camarilla R3/S3 levels from daily high/low/close,
# and volume spike (>2x average) for entry confirmation. Designed to capture continuation
# moves in the direction of the daily trend while filtering choppy markets.
# Target: 12-37 trades per year (50-150 total over 4 years).

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data for Camarilla levels, EMA trend, and volume average
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
    
    # Calculate daily Camarilla levels (R3, S3) from previous day
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    camarilla_r3 = np.full(len(close_daily), np.nan)
    camarilla_s3 = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 2:
        for i in range(1, len(close_daily)):
            # Use previous day's OHLC to calculate today's Camarilla levels
            phigh = high_daily[i-1]
            plow = low_daily[i-1]
            pclose = close_daily[i-1]
            range_val = phigh - plow
            
            camarilla_r3[i] = pclose + range_val * 1.1 / 4  # R3 = C + (H-L)*1.1/4
            camarilla_s3[i] = pclose - range_val * 1.1 / 4  # S3 = C - (H-L)*1.1/4
    
    # Calculate daily volume average for volume spike detection
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 6h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for indicators
    
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
        
        # Volume spike: current 6h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R3/S3 in direction of daily trend
            # Long when price breaks above R3 in uptrend
            long_condition = (
                close[i] > camarilla_r3_aligned[i] and   # price above R3 (breakout bullish)
                close[i] > ema34_daily_aligned[i] and    # price above EMA34 (uptrend)
                vol_spike                                # volume spike for confirmation
            )
            
            # Short when price breaks below S3 in downtrend
            short_condition = (
                close[i] < camarilla_s3_aligned[i] and   # price below S3 (breakout bearish)
                close[i] < ema34_daily_aligned[i] and    # price below EMA34 (downtrend)
                vol_spike                                # volume spike for confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or trend changes
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or trend changes
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals