#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.20) to minimize fee churn while capturing moves.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.
# Uses 4h/1d for signal direction, 1h only for entry timing.
# Session filter (08-20 UTC) to reduce noise trades.

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot levels (using prior day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Extract prior 4h bar's OHLC (we need completed prior 4h bar)
    prior_high_4h = np.roll(df_4h['high'].values, 1)
    prior_low_4h = np.roll(df_4h['low'].values, 1)
    prior_close_4h = np.roll(df_4h['close'].values, 1)
    prior_high_4h[0] = np.nan
    prior_low_4h[0] = np.nan
    prior_close_4h[0] = np.nan
    
    # Align prior 4h OHLC to 1h timeframe
    prior_high_4h_aligned = align_htf_to_ltf(prices, df_4h, prior_high_4h)
    prior_low_4h_aligned = align_htf_to_ltf(prices, df_4h, prior_low_4h)
    prior_close_4h_aligned = align_htf_to_ltf(prices, df_4h, prior_close_4h)
    
    # Calculate Camarilla levels for each 1h bar based on prior 4h bar's OHLC
    range_hl_4h = prior_high_4h_aligned - prior_low_4h_aligned
    r3_4h = prior_close_4h_aligned + range_hl_4h * 1.1 / 4
    s3_4h = prior_close_4h_aligned - range_hl_4h * 1.1 / 4
    r4_4h = prior_close_4h_aligned + range_hl_4h * 1.1 / 2
    s4_4h = prior_close_4h_aligned - range_hl_4h * 1.1 / 2
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Camarilla levels from 4h
        r3_level = r3_4h[i]
        s3_level = s3_4h[i]
        r4_level = r4_4h[i]
        s4_level = s4_4h[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion to median)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion to median)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 1d EMA50 AND volume confirmation AND in session
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3 AND price < 1d EMA50 AND volume confirmation AND in session
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals