#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Camarilla pivot levels calculated from 1d OHLC: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4.
# Long when price breaks above R3 with volume > 2.0x 20-bar average AND price > 1d EMA50.
# Short when price breaks below S3 with volume > 2.0x 20-bar average AND price < 1d EMA50.
# Exit when price returns to the 1d close (pivot point) or reverses across the opposite Camarilla level.
# Camarilla levels identify intraday support/resistance with high reversal probability.
# 1d EMA50 filters for dominant daily trend to avoid counter-trend entries.
# Volume confirmation ensures institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels: R3, S3, and pivot (close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range * 1.1 / 4  # R3 = close + 1.1*(range)*1.1/4
    s3 = close_1d - 1.1 * camarilla_range * 1.1 / 4  # S3 = close - 1.1*(range)*1.1/4
    pivot = close_1d  # 1d close as pivot point
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, uptrend (price > 1d EMA50), volume confirmation
            if (curr_high > r3_aligned[i] and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, downtrend (price < 1d EMA50), volume confirmation
            elif (curr_low < s3_aligned[i] and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price returns to pivot (1d close) OR breaks below S3 (reversal)
            if (curr_close <= pivot_aligned[i] or curr_low < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price returns to pivot (1d close) OR breaks above R3 (reversal)
            if (curr_close >= pivot_aligned[i] or curr_high > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals