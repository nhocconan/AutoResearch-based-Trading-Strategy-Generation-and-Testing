#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above R3 AND close > 4h EMA50 AND volume > 2.0x average
# Short when price breaks below S3 AND close < 4h EMA50 AND volume > 2.0x average
# Uses 4h for signal direction (trend and Camarilla levels), 1h only for entry timing
# Session filter 08-20 UTC to reduce noise trades
# Discrete sizing 0.20 targets 15-37 trades/year on 1h timeframe

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session_v1"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (using previous day's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 for 4h
    camarilla_r3_4h = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3_4h = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Camarilla (need previous day data)
    
    for i in range(start_idx, n):
        # Skip if outside trading session or HTF data not available
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_4h_aligned[i]
        curr_s3 = camarilla_s3_4h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below R3 (breakout failed)
            # 2. Price crosses below 4h EMA50 (trend change)
            if (curr_close < curr_r3 or
                curr_close < curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above S3 (breakdown failed)
            # 2. Price crosses above 4h EMA50 (trend change)
            if (curr_close > curr_s3 or
                curr_close > curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND close > 4h EMA50 AND volume confirm AND in session
            if (curr_close > curr_r3 and
                curr_close > curr_ema_50_4h and
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 AND close < 4h EMA50 AND volume confirm AND in session
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_50_4h and
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals