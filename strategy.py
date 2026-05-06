#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla R3/S3 breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above daily Camarilla R3 with EMA50 > EMA200 (uptrend) and volume > 1.5x average.
# Short when price breaks below daily Camarilla S3 with EMA50 < EMA200 (downtrend) and volume > 1.5x average.
# Camarilla levels provide precise reversal points, EMA filter ensures trend alignment, volume confirms strength.
# Target: 15-35 trades per year (60-140 over 4 years) with 0.30 position sizing.

name = "12h_1dCamarilla_R3S3_EMA50Trend_Volume"
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
    
    # Calculate daily Camarilla levels (R3, S3) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Camarilla levels: based on previous day's range
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    range_ = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (range_ * 1.1 / 4)
    camarilla_s3 = prev_close - (range_ * 1.1 / 4)
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate EMA50 and EMA200 for trend filter on 12h timeframe
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: >1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma_30)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily Camarilla R3 with uptrend and volume confirmation
            if close[i] > camarilla_r3_aligned[i] and ema_50[i] > ema_200[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below daily Camarilla S3 with downtrend and volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and ema_50[i] < ema_200[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily Camarilla S3 (reversal below support)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above daily Camarilla R3 (reversal above resistance)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals