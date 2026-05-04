#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla levels provide precise intraday support/resistance. 4h EMA50 ensures alignment with higher timeframe trend.
# Volume spike (>2x 20 EMA) confirms institutional participation. Discrete sizing 0.20 limits risk and fees.
# Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in bull/bear: trend filter prevents counter-trend entries. Uses 1h for entry timing, 4h for direction.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    close_4h = pd.Series(df_4h['close'])
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h Camarilla levels (R3, S3, R4, S4)
    # Based on previous day's high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shift to get previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align previous day's OHLC to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels: R3, S3, R4, S4
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_r3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    camarilla_s3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    camarilla_r4 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 2
    camarilla_s4 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 2
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s4[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR trend changes OR volume drops
            if (close[i] < camarilla_s3[i] or 
                close[i] < ema50_4h_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR trend changes OR volume drops
            if (close[i] > camarilla_r3[i] or 
                close[i] > ema50_4h_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals