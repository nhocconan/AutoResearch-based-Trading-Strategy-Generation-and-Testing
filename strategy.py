#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout + 1d EMA34 trend filter + volume spike
# Camarilla pivots provide intraday support/resistance levels derived from prior day's range
# 4h breakout of Camarilla R3/S3 levels with 1d EMA34 trend filter and volume confirmation
# Target: 15-35 trades/year (60-140 total over 4 years) to stay within fee drag limits
# Works in bull/bear via trend filter - only take longs in uptrend, shorts in downtrend

name = "1h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 4h and 1d calculations
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivots from prior 4h bar's range
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 for breakout entries
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate prior 4h bar's range for Camarilla levels
    range_4h = h_4h - l_4h
    camarilla_r3 = c_4h + 1.1 * range_4h
    camarilla_s3 = c_4h - 1.1 * range_4h
    
    # Align Camarilla levels to 1h timeframe (available after 4h bar closes)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price crosses below 1d EMA34 (trend change)
            # 2. Price re-enters Camarilla range (breakout failed)
            if (curr_close < curr_ema_34_1d or
                curr_close < curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price crosses above 1d EMA34 (trend change)
            # 2. Price re-enters Camarilla range (breakout failed)
            if (curr_close > curr_ema_34_1d or
                curr_close > curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + above 1d EMA34 + volume confirm + in session
            if (curr_close > curr_r3 and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Camarilla S3 + below 1d EMA34 + volume confirm + in session
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals