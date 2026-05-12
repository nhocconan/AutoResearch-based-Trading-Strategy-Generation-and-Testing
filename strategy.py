#!/usr/bin/env python3
# 1d_1w_Camarilla_R3_S3_Breakout_TrendVol_v2
# Hypothesis: Daily breakouts from weekly Camarilla R3/S3 levels with weekly trend filter and volume confirmation.
# Targets 1d timeframe to reduce trade frequency (target: 10-30 trades/year) while using proven Camarilla structure.
# Uses weekly Camarilla levels derived from prior week's OHLC, filtered by weekly EMA trend and volume spike.
# Designed to work in both bull and bear markets via trend filter that aligns with weekly momentum.

name = "1d_1w_Camarilla_R3_S3_Breakout_TrendVol_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on 1d timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior week's OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/2
    # Camarilla S3 = Close - (High - Low) * 1.1/2
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    camarilla_r3 = weekly_close + (weekly_high - weekly_low) * 1.1 / 2
    camarilla_s3 = weekly_close - (weekly_high - weekly_low) * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe (wait for weekly bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R3 + volume spike + price above weekly EMA20
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + volume spike + price below weekly EMA20
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S3 and R3 OR closes below weekly EMA20
            if (close[i] > camarilla_s3_aligned[i] and close[i] < camarilla_r3_aligned[i]) or \
               close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S3 and R3 OR closes above weekly EMA20
            if (close[i] > camarilla_s3_aligned[i] and close[i] < camarilla_r3_aligned[i]) or \
               close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals