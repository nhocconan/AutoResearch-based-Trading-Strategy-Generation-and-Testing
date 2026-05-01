#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Camarilla R3/S3 levels act as intraday support/resistance; breakouts with volume and HTF trend alignment capture momentum.
# Works in bull (buy breakouts above R3) and bear (sell breakdowns below S3) markets.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag on 12h timeframe.

name = "12h_Camarilla_R3S3_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Calculate Camarilla levels from previous 1d bar (requires prior day's OHLC)
        # Camarilla levels are calculated from previous day's range
        if i < start_idx + 1:  # need at least one prior day
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (1d data)
        # Since we're on 12h timeframe, we need to reference completed 1d bars
        # We'll use the 1d data we already loaded
        # Find index of previous completed 1d bar
        prev_1d_idx = len(df_1d) - 1  # most recent completed 1d bar
        # But we need to ensure we're using the 1d bar that completed before current 12h bar
        # align_htf_to_ltf already handles this delay, so we can use the aligned arrays
        
        # Instead, calculate Camarilla levels using the 1d data with proper alignment
        # We need the prior completed 1d bar's OHLC
        # We'll compute Camarilla levels for each 1d bar and align to 12h
        
        # Calculate Camarilla levels for 1d data
        # Camarilla formulas:
        # H4 = Close + 1.1*(High-Low)/2
        # H3 = Close + 1.1*(High-Low)/4
        # H2 = Close + 1.1*(High-Low)/6
        # H1 = Close + 1.1*(High-Low)/12
        # L1 = Close - 1.1*(High-Low)/12
        # L2 = Close - 1.1*(High-Low)/6
        # L3 = Close - 1.1*(High-Low)/4
        # L4 = Close - 1.1*(High-Low)/2
        # R3 and S3 are the same as H3 and L3
        
        # We'll compute these for the 1d data and align
        # But we need to do this outside the loop for efficiency
        
        # Move Camarilla calculation outside loop
        pass  # We'll restructure
    
    # Restructure: calculate Camarilla levels for 1d data once
    # Calculate typical price and range for Camarilla
    hlc_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Actually, Camarilla uses OHLC directly
    # Let's compute properly
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3
    r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 4
    s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Now load 4h data for volume confirmation (to get more frequent volume data)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_r3 = r3_1d_aligned[i]
        curr_s3 = s3_1d_aligned[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        
        # Volume confirmation: current 12h volume > 2.0x 20-period average from 4h data
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above R3 AND price > 1d EMA50 AND volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 AND price < 1d EMA50 AND volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < S3 (reversal) OR price < 1d EMA50 (trend violation)
            if (curr_close < curr_s3 or 
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > R3 (reversal) OR price > 1d EMA50 (trend violation)
            if (curr_close > curr_r3 or 
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals