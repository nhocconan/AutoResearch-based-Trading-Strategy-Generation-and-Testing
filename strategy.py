#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout on 4h with 12h EMA50 trend filter and volume spike (>2x average volume). Uses discrete position sizing (0.25) to minimize fee churn. Exits when price retests the broken Camarilla level (R3 for longs, S3 for shorts) or reverses across the 12h EMA50. Works in both bull and bear markets by following the 12h trend direction, confirmed by volume to avoid false breakouts. Dynamic exit reduces whipsaw vs fixed ATR stop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA, volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for HTF trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    camarilla_s3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (1 bar delay for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA, 20 for volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_12h_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r3_val) or 
            np.isnan(s3_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2x average volume (strong breakout)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above Camarilla R3 with 12h uptrend and volume confirmation
        long_condition = (close_val > r3_val) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S3 with 12h downtrend and volume confirmation
        short_condition = (close_val < s3_val) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: 
        # Long exit: price retests or breaks below Camarilla R3 (failed breakout) OR closes below 12h EMA (trend change)
        long_exit = (position == 1 and (close_val <= r3_val or close_val < ema_val))
        # Short exit: price retests or breaks above Camarilla S3 (failed breakout) OR closes above 12h EMA (trend change)
        short_exit = (position == -1 and (close_val >= s3_val or close_val > ema_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0