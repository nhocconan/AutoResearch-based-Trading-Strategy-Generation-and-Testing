#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Uses tight volume threshold (2.5x average) to limit trades to ~80 total over 4 years.
# Only enters when price breaks Camarilla R3 (long) or S3 (short) with volume confirmation and 4h EMA50 trend alignment.
# Designed for low trade frequency to avoid fee drag. Works in bull/bear via 4h EMA50 trend filter.
# Session filter (08-20 UTC) reduces noise trades. Position size 0.20 to manage drawdown.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels using previous 4h bar (completed)
    typical_price = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3.0
    range_hl = df_4h['high'].values - df_4h['low'].values
    
    # Camarilla levels
    R3 = typical_price + (range_hl * 1.1 / 4.0)
    S3 = typical_price - (range_hl * 1.1 / 4.0)
    R4 = typical_price + (range_hl * 1.1 / 2.0)
    S4 = typical_price - (range_hl * 1.1 / 2.0)
    
    # Align to 1h timeframe with proper delay (wait for 4h bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    R4_aligned = align_htf_to_ltf(prices, df_4h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_4h, S4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        
        # Volume confirmation: volume > 2.5x 20-period average (tight threshold to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, 4h EMA50 uptrend, volume spike confirmation
            if (curr_close > R3_aligned[i] and 
                curr_close > curr_ema_50_4h and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below S3, 4h EMA50 downtrend, volume spike confirmation
            elif (curr_close < S3_aligned[i] and 
                  curr_close < curr_ema_50_4h and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below S3 or reverses below entry
            if curr_close < S3_aligned[i] or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above R3 or reverses above entry
            if curr_close > R3_aligned[i] or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals