#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels (R3/S3) with volume confirmation and ADX trend filter
# Long when price breaks above S3 with ADX > 25 and volume > 1.5x average
# Short when price breaks below R3 with ADX > 25 and volume > 1.5x average
# Uses daily pivot levels for strong support/resistance, ADX to filter ranging markets, volume to confirm breakout strength
# Target: 15-30 trades per year (60-120 over 4 years) with 0.25 position sizing

name = "12h_1dCamarilla_R3S3_ADXVol_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels (R3, S3) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Camarilla levels: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    r3 = prev_close + (range_hl * 1.1 / 2)
    s3 = prev_close - (range_hl * 1.1 / 2)
    
    # Align daily Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Calculate daily ADX for trend filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range and Directional Movement
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = df_1d['high'].diff()
    minus_dm = df_1d['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above S3 with trend and volume confirmation
            if close[i] > s3_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below R3 with trend and volume confirmation
            elif close[i] < r3_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (trend reversal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (trend reversal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals