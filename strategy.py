#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels identify intraday support/resistance. Breakout of R3 (resistance 3) or S3 (support 3)
# with 4h EMA50 trend alignment captures momentum moves. Volume spike confirms conviction.
# Session filter (08-20 UTC) reduces noise. Designed for 15-30 trades/year on 1h to minimize fee drag.
# Works in bull markets via breakouts and bear markets via faded extremes at key levels.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        # Only calculate at 00:00 UTC bars (start of new day)
        if i == 0 or open_time[i].date() != open_time[i-1].date():
            if i >= 1:
                prev_high = high[i-1]
                prev_low = low[i-1]
                prev_close = close[i-1]
                rang = prev_high - prev_low
                camarilla_r3[i] = prev_close + rang * 1.1 / 4
                camarilla_s3[i] = prev_close - rang * 1.1 / 4
            else:
                camarilla_r3[i] = np.nan
                camarilla_s3[i] = np.nan
        else:
            camarilla_r3[i] = camarilla_r3[i-1]
            camarilla_s3[i] = camarilla_s3[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Camarilla breakout conditions
        breakout_long = close[i] > camarilla_r3[i]
        breakout_short = close[i] < camarilla_s3[i]
        
        if position == 0:
            # Long: breakout above R3 in 4h uptrend with volume spike
            if breakout_long and ema_50_4h_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 in 4h downtrend with volume spike
            elif breakout_short and ema_50_4h_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: close below R3 or loses 4h uptrend
            if close[i] < camarilla_r3[i] or ema_50_4h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: close above S3 or loses 4h downtrend
            if close[i] > camarilla_s3[i] or ema_50_4h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals