#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# In ranging markets (CHOP > 50), we mean-revert at Camarilla R3/S3 levels with volume spike.
# In trending markets (CHOP < 50), we breakout above R3 or below S3 with volume spike and 12h EMA50 trend alignment.
# Designed to work in both bull and bear markets by adapting to regime with tight entry conditions.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 1d data for Camarilla levels and CHOP regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d CHOP regime: CHOP > 50 = ranging (mean revert), CHOP < 50 = trending (trend follow)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first TR
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1d - lowest_low_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_1d = 100 * (np.log10(atr_1d * np.sqrt(14) / chop_denom) / np.log10(10))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels from previous 1d bar (use shift to avoid look-ahead)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_range = 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_r3 = prev_close_1d + camarilla_range
    camarilla_s3 = prev_close_1d - camarilla_range
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        chop_val = chop_1d_aligned[i]
        ema_trend = ema_50_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(chop_val) or np.isnan(ema_trend) or np.isnan(r3_level) or np.isnan(s3_level):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime-based entry conditions
        if chop_val > 50:  # Ranging market: mean reversion at Camarilla levels
            # Long: price crosses above S3 with volume spike (mean reversion up)
            long_entry = (close[i] > s3_level) and (close[i-1] <= s3_level) and vol_spike
            # Short: price crosses below R3 with volume spike (mean reversion down)
            short_entry = (close[i] < r3_level) and (close[i-1] >= r3_level) and vol_spike
        else:  # Trending market: breakout with trend alignment
            # Long: price breaks above R3 with volume spike and above EMA50
            long_entry = (close[i] > r3_level) and (close[i-1] <= r3_level) and vol_spike and (close[i] > ema_trend)
            # Short: price breaks below S3 with volume spike and below EMA50
            short_entry = (close[i] < s3_level) and (close[i-1] >= s3_level) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below S3 (mean reversion) or below EMA50 (trend change)
            if close[i] < s3_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above R3 (mean reversion) or above EMA50 (trend change)
            if close[i] > r3_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals