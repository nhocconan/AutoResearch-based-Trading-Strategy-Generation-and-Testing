#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from daily structure for institutional breakout levels
# 1w EMA50 filter ensures trading with primary weekly trend direction
# Volume spike (>2.0x 20-bar avg) confirms institutional participation
# Designed for low-frequency, high-conviction trades (target: 12-37/year) that work in both bull/bear via trend alignment
# Exits on opposite Camarilla level (R3/S3) touch or close back inside the daily range

name = "12h_Camarilla_R3S3_1wEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) on daily data
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * camarilla_range * 1.1 / 4
    s3_level = close_1d - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (completed daily candles only)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_level)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r3 = r3_12h[i]
        s3 = s3_12h[i]
        ema_50 = ema_50_12h[i]
        daily_high = daily_high_aligned[i]
        daily_low = daily_low_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above R3 AND price > 1w EMA50 (uptrend) AND volume spike
            if price > r3 and price > ema_50 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price breaks below S3 AND price < 1w EMA50 (downtrend) AND volume spike
            elif price < s3 and price < ema_50 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on price touching S3 (opposite level) or back inside daily range
            # Exit on price < S3 (opposite Camarilla level) or price < daily_low (failed breakout)
            if price < s3 or price < daily_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on price touching R3 (opposite level) or back inside daily range
            # Exit on price > R3 (opposite Camarilla level) or price > daily_high (failed breakout)
            if price > r3 or price > daily_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals