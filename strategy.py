#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance. Breakouts above R3
# or below S3 with volume confirmation and 4h EMA50 trend alignment provide high-probability
# trend continuation entries. Designed for 1h timeframe to target 15-37 trades/year (60-150 total)
# with discrete sizing (0.20). Uses 4h for trend direction and 1h only for entry timing precision.
# Session filter (08-20 UTC) reduces noise. Works in bull markets by buying breakouts in uptrends
# and in bear markets by selling breakdowns in downtrends, avoiding whipsaws during ranging periods.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume"
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily Camarilla pivot points (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 = close ± (1.1/12)*(high-low)
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (1.1/12) * range_1d
    camarilla_s3 = close_1d - (1.1/12) * range_1d
    
    # Align Camarilla levels to 1h timeframe (each 1d level applies to 24 consecutive 1h bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 1.5x 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0 and in_session:
            # Long: price breaks above Camarilla R3 + volume confirmation + price above 4h EMA50 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and volume_confirmed and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + volume confirmation + price below 4h EMA50 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 (reversal) OR price below 4h EMA50 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 (reversal) OR price above 4h EMA50 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Outside session or no signal: maintain flat or current position
            if position != 0:
                signals[i] = 0.20 if position == 1 else -0.20
            else:
                signals[i] = 0.0
    
    return signals