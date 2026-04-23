#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA Trend Filter and Volume Spike
- Uses tighter Camarilla R3/S3 levels (more sensitive than R4/S4) for earlier entry
- 12h EMA(50) trend filter ensures alignment with intermediate-term trend
- Volume > 1.8x 20-period average confirms breakout momentum (slightly looser than 2.0x)
- Designed for 4h timeframe targeting 30-50 trades/year (120-200 over 4 years)
- Works in bull markets via breakouts with trend, in bear markets via mean reversion at strong levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h timeframe (completed 12h bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA needs 50 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from 12h EMA
        # Uptrend: price above EMA50, Downtrend: price below EMA50
        # Use 12h close price for trend determination (need to align 12h close)
        close_12h = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema_50_aligned[i]
        downtrend = close_12h_aligned[i] < ema_50_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume spike
        # Long: price breaks above Camarilla R3 + uptrend + volume spike
        # Short: price breaks below Camarilla S3 + downtrend + volume spike
        long_signal = (close[i] > camarilla_r3_aligned[i] and 
                      uptrend and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s3_aligned[i] and 
                       downtrend and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend turns down or price breaks below Camarilla S3
                if (not uptrend or 
                    close[i] < camarilla_s3_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend turns up or price breaks above Camarilla R3
                if (not downtrend or 
                    close[i] > camarilla_r3_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0