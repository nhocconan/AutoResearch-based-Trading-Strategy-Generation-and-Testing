#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 1w EMA200 trend filter and volume confirmation
# Uses Camarilla pivot levels from previous 6h bar for intraday structure
# Only trade breakouts above R3 or below S3 in direction of 1w EMA200 trend
# Volume spike (2.0x 24-period average) confirms institutional participation
# Designed for low frequency (target: 75-150 total trades over 4 years = 19-37/year) to minimize fee drag
# Works in bull markets via buying R3 breakouts in uptrends and bear markets via selling S3 breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Uses 1w EMA200 for strong trend filter to avoid whipsaws.

name = "6h_Camarilla_R3S3_Breakout_1wEMA200_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA200
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Need prior 6h bar for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Previous 6h bar's OHLC for Camarilla pivot calculation
        y_high = high[i-1]
        y_low = low[i-1]
        y_close = close[i-1]
        
        # Calculate Camarilla levels
        pivot = (y_high + y_low + y_close) / 3
        range_hl = y_high - y_low
        
        # Camarilla R3 and S3 (most significant levels)
        r3 = y_close + range_hl * 1.1 / 2
        s3 = y_close - range_hl * 1.1 / 2
        
        # Volume confirmation: volume > 2.0x 24-period average (4 days)
        vol_ma_24 = np.mean(volume[max(0, i-24):i])
        volume_spike = volume[i] > (2.0 * vol_ma_24)
        
        curr_close = close[i]
        curr_ema_200_1w = ema_200_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above R3 AND above 1w EMA200 (strong uptrend)
                if curr_close > r3 and curr_close > curr_ema_200_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 AND below 1w EMA200 (strong downtrend)
                elif curr_close < s3 and curr_close < curr_ema_200_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or below 1w EMA200
            if curr_close < s3 or curr_close < curr_ema_200_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or above 1w EMA200
            if curr_close > r3 or curr_close > curr_ema_200_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals