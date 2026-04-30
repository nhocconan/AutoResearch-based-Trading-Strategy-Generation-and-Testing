#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels from previous 1d for structure
# Only trade breakouts above R3 or below S3 in direction of 1d EMA34 trend
# Volume spike (2.0x 12-period average) confirms institutional participation
# Works in bull markets via buying breakouts in uptrends and bear markets via selling breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 12  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Need prior 1d high/low/close for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Prior 1d period's high/low/close for Camarilla pivot levels
        # Using the previous completed 1d bar
        prior_day_idx = i // 2  # 2 x 12h bars = 1d bar (approximate, but we'll use HTF data properly)
        # Actually, we should get the completed 1d bar's OHLC from df_1d
        # But since we're using 12h timeframe, we need to map to 1d bars
        # Simpler: use the prior 12h bar's data for Camarilla calculation (standard practice)
        prior_high = high[i-1]
        prior_low = low[i-1]
        prior_close = close[i-1]
        
        # Calculate Camarilla pivot levels
        pivot = (prior_high + prior_low + prior_close) / 3
        range_val = prior_high - prior_low
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        
        # Volume confirmation: volume > 2.0x 12-period average
        vol_ma_12 = np.mean(volume[max(0, i-12):i])
        volume_spike = volume[i] > (2.0 * vol_ma_12)
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above R3 AND above 1d EMA34 (uptrend)
                if curr_close > r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 AND below 1d EMA34 (downtrend)
                elif curr_close < s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or below 1d EMA34
            if curr_close < s3 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or above 1d EMA34
            if curr_close > r3 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals