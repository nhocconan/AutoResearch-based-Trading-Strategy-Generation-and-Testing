#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level with 4h EMA50 uptrend and volume > 1.5x 20-period volume EMA
# Short when price breaks below Camarilla S3 level with 4h EMA50 downtrend and volume > 1.5x 20-period volume EMA
# Uses 4h HTF for trend to reduce whipsaw vs shorter HTF, targeting 15-37 trades/year on 1h.
# Volume spike filter (1.5x) is moderate to avoid overtrading. Camarilla levels provide clear pivot structure.
# Session filter (08-20 UTC) reduces noise trades outside active hours.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "1h_Camarilla_R3S3_4hEMA50_Trend_VolumeSpike"
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
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # R3 = close + 1.125*(high - low), S3 = close - 1.125*(high - low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].values
    
    camarilla_r3 = close_4h_prev + 1.125 * (high_4h - low_4h)
    camarilla_s3 = close_4h_prev - 1.125 * (high_4h - low_4h)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 4h uptrend AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and  # 4h uptrend
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 4h downtrend AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and  # 4h downtrend
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 4h trend turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 4h trend turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals