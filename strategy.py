#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Uses 4h for signal direction (EMA50 trend) and 1h only for entry timing precision
# Long when 1h close > R3 AND 4h price > 4h EMA50 AND 1h volume > 2.0x 20-bar avg
# Short when 1h close < S3 AND 4h price < 4h EMA50 AND 1h volume > 2.0x 20-bar avg
# Exit on opposite Camarilla level touch (long exit at S3, short exit at R3)
# Uses discrete position sizing (0.20) to minimize fee drag. Target: 15-37 trades/year on 1h.
# Session filter: 08-20 UTC to reduce noise trades
# Camarilla levels from previous day provide institutional support/resistance
# EMA50 filters counter-trend moves, volume spike confirms participation

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla calculation (previous day's levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate typical price for 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    typical_price_1d_values = typical_price_1d.values
    # Previous day's typical price
    prev_typical = pd.Series(typical_price_1d_values).shift(1).values
    # Daily range for 1d data
    daily_range_1d = df_1d['high'] - df_1d['low']
    daily_range_1d_values = daily_range_1d.values
    # Previous day's range
    prev_range = pd.Series(daily_range_1d_values).shift(1).values
    
    # Camarilla levels (based on previous day)
    R3 = prev_typical + (prev_range * 1.1 / 4)
    S3 = prev_typical - (prev_range * 1.1 / 4)
    R4 = prev_typical + (prev_range * 1.1 / 2)
    S4 = prev_typical - (prev_range * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: >2.0x 20-bar average volume on 1h
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 needs 50 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_4h_aligned[i]
        r3_level = R3_aligned[i]
        s3_level = S3_aligned[i]
        r4_level = R4_aligned[i]
        s4_level = S4_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when close > R3 AND price > 4h EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when close < S3 AND price < 4h EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when close < S3 (opposite level)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when close > R3 (opposite level)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals