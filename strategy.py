#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses 4h timeframe for signal generation with Camarilla pivot levels from daily data
# 12h EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Camarilla levels provide precise intraday support/resistance, effective in both trending and ranging markets
# Trend filter prevents false signals during weak trends, volume spike confirms validity

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        # Need previous day's data - use index i-1 for 1d data alignment
        if i < 24:  # Need at least 24 hours of 1h data (approximate)
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC from 1d data (already aligned)
        # We'll use the 1d data that ended at or before current bar
        day_idx = i // 24  # Approximate day index (assuming 24*1h bars per day)
        if day_idx >= len(df_1d):
            signals[i] = 0.0
            continue
            
        prev_high = df_1d['high'].iloc[day_idx]
        prev_low = df_1d['low'].iloc[day_idx]
        prev_close = df_1d['close'].iloc[day_idx]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_r3 = prev_close + (range_val * 1.1 / 4)
        camarilla_s3 = prev_close - (range_val * 1.1 / 4)
        camarilla_r4 = prev_close + (range_val * 1.1 / 2)
        camarilla_s4 = prev_close - (range_val * 1.1 / 2)
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > R3 + volume confirm + price > 12h EMA50 (uptrend)
            if close[i] > camarilla_r3 and volume_confirm[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 + volume confirm + price < 12h EMA50 (downtrend)
            elif close[i] < camarilla_s3 and volume_confirm[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < S3 (reversal to support) or bearish trend
            if close[i] < camarilla_s3 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > R3 (reversal to resistance) or bullish trend
            if close[i] > camarilla_r3 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals