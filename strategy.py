#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Uses tight volume threshold (2.5x average) to limit trades to ~120 total over 4 years.
# Only enters when price breaks 12h Camarilla R3 (short) or S3 (long) levels with volume confirmation and 1d EMA50 trend alignment.
# Designed for low trade frequency (<200 total 12h trades) to avoid fee drag. Works in bull/bear via 1d EMA50 trend filter.

name = "12h_Camarilla_R3S3_1dEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(1, 50) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume = volume[i]
        
        # Calculate 12h Camarilla levels for R3 and S3 using only completed 12h bars
        # Need at least one completed 12h bar (lookback of 1 bar) to calculate
        if i >= 1:
            # For 12h timeframe, we use the previous completed 12h bar's OHLC
            # Since we're on 12h timeframe, we can use i-1 as the previous bar
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Camarilla levels calculation
            range_val = prev_high - prev_low
            camarilla_s3 = prev_close + (range_val * 1.1 / 4)  # S3 level
            camarilla_r3 = prev_close - (range_val * 1.1 / 4)  # R3 level
        else:
            camarilla_s3 = np.nan
            camarilla_r3 = np.nan
        
        # Volume confirmation: volume > 2.5x 20-period average (tight threshold to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = curr_volume > (2.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla S3, 1d EMA50 uptrend, volume spike confirmation
            if (curr_close > camarilla_s3 and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla R3, 1d EMA50 downtrend, volume spike confirmation
            elif (curr_close < camarilla_r3 and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit if price breaks below Camarilla S3 or 1d EMA50 turns down
            if (curr_close < camarilla_s3 or 
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3 or 1d EMA50 turns up
            if (curr_close > camarilla_r3 or 
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals