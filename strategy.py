#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above R3 AND close > 12h EMA50 AND volume > 2.0x average
# Short when price breaks below S3 AND close < 12h EMA50 AND volume > 2.0x average
# Uses discrete sizing (0.25) and tight entry conditions to target 20-50 trades/year.
# Camarilla levels provide institutional support/resistance; 12h EMA50 filters trend; volume confirms conviction.
# Timeframe: 4h (primary), HTF: 12h for EMA50 trend.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels (using previous 12h bar's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 for 12h
    camarilla_r3_12h = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3_12h = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Camarilla (need previous 12h bar data)
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_12h_aligned[i]) or np.isnan(camarilla_s3_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_r3 = camarilla_r3_12h_aligned[i]
        curr_s3 = camarilla_s3_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below R3 (breakout failed)
            # 2. Price crosses below 12h EMA50 (trend change)
            if (curr_close < curr_r3 or
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above S3 (breakdown failed)
            # 2. Price crosses above 12h EMA50 (trend change)
            if (curr_close > curr_s3 or
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND close > 12h EMA50 AND volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_50_12h and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND close < 12h EMA50 AND volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_50_12h and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals