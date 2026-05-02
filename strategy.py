#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla R3/S3 levels from previous 4h bar for breakout entries on 1h timeframe.
# 4h EMA50 provides trend filter to avoid counter-trend trades.
# Volume confirmation (2.0x 20-period average) ensures institutional participation.
# Session filter (08-20 UTC) reduces noise trades.
# Designed for low trade frequency (~60-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets via breakouts with trend, in bear via avoidance of false breakouts.
# Target: BTC/ETH/SOL with Sharpe > 0 on both train and test.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop for HTF calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla levels from previous 4h bar (yesterday's OHLC)
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # True range for Camarilla calculation
    tr = np.maximum(prev_high - prev_low, 
                    np.maximum(np.abs(prev_high - prev_close), 
                               np.abs(prev_low - prev_close)))
    
    # Camarilla R3, S3 levels (strong breakout levels)
    camarilla_r3 = prev_close + (tr * 1.1 / 4)
    camarilla_s3 = prev_close - (tr * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA, volume MA, and pivot calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 4h EMA50 + volume confirm + session
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 4h EMA50 + volume confirm + session
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 (strong reversal signal)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 (strong reversal signal)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals