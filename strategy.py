# 6h Camarilla R3/S3 Breakout with Weekly Trend Filter
# Hypothesis: Camarilla pivot levels from daily data provide strong support/resistance.
# Breakouts beyond R3/S3 with weekly trend alignment capture momentum moves.
# Weekly trend filter avoids counter-trend trades in ranging markets.
# Works in bull/bear by aligning with higher timeframe direction.
# Target: 50-150 total trades over 4 years (12-37/year).
# Position size: 0.25

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly trend using EMA(34) on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Precompute Camarilla levels for each day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_r4 = np.zeros(len(df_1d))
    camarilla_s4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 1:  # Need previous day's data
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_r4[i] = np.nan
            camarilla_s4[i] = np.nan
        else:
            # Use previous day's OHLC to avoid look-ahead
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            rang = prev_high - prev_low
            camarilla_r3[i] = prev_close + (rang * 1.1 / 4)
            camarilla_s3[i] = prev_close - (rang * 1.1 / 4)
            camarilla_r4[i] = prev_close + (rang * 1.1 / 2)
            camarilla_s4[i] = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 6h volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume above average
        volume_filter = vol_ma_20[i] > 0 and volume[i] > vol_ma_20[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Breakout conditions
        breakout_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_s3 = close[i] < camarilla_s3_aligned[i]
        breakout_r4 = close[i] > camarilla_r4_aligned[i]
        breakout_s4 = close[i] < camarilla_s4_aligned[i]
        
        # Long conditions: weekly uptrend + volume + breakout above R3
        long_condition = (weekly_uptrend and 
                         volume_filter and 
                         breakout_r3)
        
        # Short conditions: weekly downtrend + volume + breakdown below S3
        short_condition = (weekly_downtrend and 
                          volume_filter and 
                          breakout_s3)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (breakout_s3 or not weekly_uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_r3 or not weekly_downtrend):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0