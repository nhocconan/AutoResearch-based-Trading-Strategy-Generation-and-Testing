#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_4hTrend_Volume
Hypothesis: On 1h timeframe, buy when price breaks above Camarilla R3 level with 4h uptrend (close > EMA34) and volume confirmation; sell when breaks below S3 level with 4h downtrend (close < EMA34) and volume confirmation. Uses 4h EMA34 for trend filter to avoid whipsaws and volume spike for confirmation. Designed for 1h timeframe with expected 60-150 trades over 4 years to minimize fee drag while capturing trends in both bull and bear markets.
"""
name = "1h_Camarilla_R3S3_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    # Camarilla levels use previous bar's data
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Calculate Camarilla levels for each 4h bar
    # R3 = C + (H-L)*1.1/2
    # S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume filter: current volume > 1.5 * 20-period average volume (1h timeframe)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + 4h uptrend + volume filter
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + 4h downtrend + volume filter
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level
            if position == 1:
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals