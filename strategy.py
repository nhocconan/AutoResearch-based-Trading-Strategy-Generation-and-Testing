#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d EMA50 trend filter
# Long when: price breaks above R3, volume > 2x 20-period 4h average, and close > 1d EMA50
# Short when: price breaks below S3, volume > 2x 20-period 4h average, and close < 1d EMA50
# Exit when price returns to Camarilla R3/S3 level (mean reversion)
# Uses Camarilla levels from 1h for structure, with 4h volume and 1d trend filters to reduce false breakouts.
# Timeframe: 1h, HTF: 4h (volume) and 1d (EMA, Camarilla levels). Target: 60-150 total trades over 4 years (15-37/year).
# Session filter: 08-20 UTC to avoid low-liquidity hours.

name = "1h_Camarilla_R3S3_Breakout_4hVolume_1dEMA50"
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
    open_price = prices['open'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Calculate volume confirmation on 4h (done once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    if len(volume_4h) >= 20:
        vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
        volume_spike_4h = volume_4h > (2.0 * vol_ma_20_4h)
    else:
        volume_spike_4h = np.zeros(len(df_4h), dtype=bool)
    
    # Align 4h volume spike to 1h timeframe
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * rang * 1.1 / 4
        camarilla_s3 = prev_close - 1.1 * rang * 1.1 / 4
    else:
        camarilla_r3 = np.full(len(close_1d), np.nan)
        camarilla_s3 = np.full(len(close_1d), np.nan)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_4h_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume spike, and above 1d EMA50
            if (close[i] > camarilla_r3_aligned[i] and 
                open_price[i] <= camarilla_r3_aligned[i] and  # Ensure breakout happens on this bar
                volume_spike_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3, volume spike, and below 1d EMA50
            elif (close[i] < camarilla_s3_aligned[i] and 
                  open_price[i] >= camarilla_s3_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_spike_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 (mean reversion)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above S3 (mean reversion)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals