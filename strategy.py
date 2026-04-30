#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from previous 4h bar for intraday structure
# Only trade breakouts above R3 or below S3 in direction of 12h EMA50 trend
# Volume spike (1.8x 20-period average) confirms institutional participation
# Works in bull markets via buying R3 breakouts in uptrends and bear markets via selling S3 breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 100-200 total trades over 4 years (25-50/year).

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Need prior 4h bar for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Previous 4h bar's OHLC for Camarilla pivot calculation
        y_high = high[i-1]
        y_low = low[i-1]
        y_close = close[i-1]
        
        # Calculate Camarilla levels
        pivot = (y_high + y_low + y_close) / 3
        range_hl = y_high - y_low
        
        # Camarilla R3 and S3 (most significant levels)
        r3 = y_close + range_hl * 1.1 / 2
        s3 = y_close - range_hl * 1.1 / 2
        
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above R3 AND above 12h EMA50 (uptrend)
                if curr_close > r3 and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 AND below 12h EMA50 (downtrend)
                elif curr_close < s3 and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or below 12h EMA50
            if curr_close < s3 or curr_close < curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or above 12h EMA50
            if curr_close > r3 or curr_close > curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals