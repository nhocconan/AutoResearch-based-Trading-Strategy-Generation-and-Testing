# 6h_Camarilla_Pivot_1dTrend_VolumeBreakout
# Hypothesis: Uses daily Camarilla pivot levels from the previous day to identify
# key support/resistance zones. Enters long at S3 breakout with volume confirmation
# when daily trend is bullish, and shorts at R3 breakdown with volume confirmation
# when daily trend is bearish. The 6h timeframe provides sufficient filtering to
# avoid overtrading while capturing meaningful breakouts. Works in both bull and
# bear markets by following the daily trend direction for bias.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_Pivot_1dTrend_VolumeBreakout"
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
    
    # Get daily data for Camarilla pivots and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.25
    # R2 = Close + (High - Low) * 1.166
    # R1 = Close + (High - Low) * 1.083
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High - Low) * 1.083
    # S2 = Close - (High - Low) * 1.166
    # S3 = Close - (High - Low) * 1.25
    # S4 = Close - (High - Low) * 1.5
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Previous day's values for pivot calculation
    high_prev = np.roll(high_daily, 1)
    low_prev = np.roll(low_daily, 1)
    close_prev = np.roll(close_daily, 1)
    # First day has no previous day
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Calculate pivot levels
    r4 = close_prev + (high_prev - low_prev) * 1.5
    r3 = close_prev + (high_prev - low_prev) * 1.25
    r2 = close_prev + (high_prev - low_prev) * 1.166
    r1 = close_prev + (high_prev - low_prev) * 1.083
    pp = (high_prev + low_prev + close_prev) / 3
    s1 = close_prev - (high_prev - low_prev) * 1.083
    s2 = close_prev - (high_prev - low_prev) * 1.166
    s3 = close_prev - (high_prev - low_prev) * 1.25
    s4 = close_prev - (high_prev - low_prev) * 1.5
    
    # Calculate daily volume average for volume breakout
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 6h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
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
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current 6h volume > 2x 20-period average of daily volume
        vol_breakout = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_breakout = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: follow daily EMA trend with Camarilla breakout and volume confirmation
            # Bullish trend: price above daily EMA34
            bullish_trend = close[i] > ema34_daily_aligned[i]
            # Bearish trend: price below daily EMA34
            bearish_trend = close[i] < ema34_daily_aligned[i]
            
            # Long when price breaks above S3 with volume confirmation in bullish trend
            long_condition = (
                bullish_trend and
                close[i] > s3_aligned[i] and   # price above S3 (breakout)
                vol_breakout                   # volume breakout for entry
            )
            
            # Short when price breaks below R3 with volume confirmation in bearish trend
            short_condition = (
                bearish_trend and
                close[i] < r3_aligned[i] and   # price below R3 (breakdown)
                vol_breakout                   # volume breakout for entry
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below S3 or trend turns bearish
            if close[i] < s3_aligned[i] or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above R3 or trend turns bullish
            if close[i] > r3_aligned[i] or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals