#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot levels (S3/R3) with volume confirmation and Choppiness index regime filter
# Long when price crosses above S3 level with volume > 1.5x 20-period average and CHOP > 61.8 (ranging market)
# Short when price crosses below R3 level with volume > 1.5x 20-period average and CHOP > 61.8 (ranging market)
# Uses daily Camarilla levels for mean reversion in ranging conditions, volume for confirmation, CHOP to avoid trending markets
# Designed to work in both bull and bear markets by exploiting mean reversion in ranges
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "4h_1dCamarilla_S3R3_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels (S3/R3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # S3 and R3 levels
    s3 = pivot - 1.1 * range_ * 1.166
    r3 = pivot + 1.1 * range_ * 1.166
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Choppiness index regime filter: CHOP > 61.8 = ranging market (mean reversion)
    # Calculate CHOP on 4h timeframe
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10((max_high - min_low) / (atr * 14)) / np.log10(14)
    chop_filter = chop > 61.8  # Ranging market condition
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S3 with volume confirmation in ranging market
            if close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R3 with volume confirmation in ranging market
            elif close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 or R3 (mean reversion complete)
            if close[i] < s3_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R3 or S3 (mean reversion complete)
            if close[i] > r3_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals