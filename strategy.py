#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Uses tight volume threshold (2.5x average) to limit trades to ~80 total over 4 years.
# Only enters when price breaks Camarilla R3 (long) or S3 (short) with volume confirmation and 1d EMA50 trend alignment.
# Designed for low trade frequency to avoid fee drag. Works in bull/bear via 1d EMA50 trend filter.

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
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
    
    start_idx = 50  # warmup for EMA50
    
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
        
        # Calculate Camarilla levels using previous 1d bar (completed)
        # We need the previous completed 1d bar's OHLC
        # Since we're on 6h timeframe, we need to get the 1d data for Camarilla calculation
        # We'll use the 1d data we already loaded
        
        # Find the index of the previous completed 1d bar in df_1d
        # We need to map current 6h bar to the 1d bar that completed before it
        # align_htf_to_ltf already handles this delay for us
        
        # For Camarilla calculation, we need the previous 1d bar's OHLC
        # We'll use the 1d data and shift by 1 to get the previous completed bar
        if len(df_1d) >= 2:
            # Get the previous completed 1d bar's OHLC
            prev_1d_idx = len(df_1d) - 2  # second to last is the previous completed bar
            # But we need to do this properly for each point in time
            
            # Instead, we'll calculate Camarilla levels for each 1d bar and then align
            # Calculate typical price for Camarilla
            typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
            range_hl = df_1d['high'].values - df_1d['low'].values
            
            # Camarilla levels
            R3 = typical_price + (range_hl * 1.1 / 4.0)
            S3 = typical_price - (range_hl * 1.1 / 4.0)
            R4 = typical_price + (range_hl * 1.1 / 2.0)
            S4 = typical_price - (range_hl * 1.1 / 2.0)
            
            # Align to 6h timeframe with proper delay (wait for 1d bar to close)
            R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
            S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
            R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
            S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
        else:
            R3_aligned = np.full(n, np.nan)
            S3_aligned = np.full(n, np.nan)
            R4_aligned = np.full(n, np.nan)
            S4_aligned = np.full(n, np.nan)
        
        # Volume confirmation: volume > 2.5x 20-period average (tight threshold to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, 1d EMA50 uptrend, volume spike confirmation
            if (curr_close > R3_aligned[i] and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below S3, 1d EMA50 downtrend, volume spike confirmation
            elif (curr_close < S3_aligned[i] and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below S3 or reverses below entry
            if curr_close < S3_aligned[i] or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above R3 or reverses above entry
            if curr_close > R3_aligned[i] or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals