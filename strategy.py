#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 Breakout with 4h EMA200 trend filter and volume confirmation
# Uses Camarilla pivot levels from previous 1h bar for intraday structure
# Only trade breakouts above R1 or below S1 in direction of 4h EMA200 trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Session filter (08-20 UTC) to avoid low-liquidity hours
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R1S1_Breakout_4hEMA200_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop (MTF Rule #1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA200
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Need prior 1h bar for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Previous 1h bar's OHLC for Camarilla pivot calculation
        y_high = high[i-1]
        y_low = low[i-1]
        y_close = close[i-1]
        
        # Calculate Camarilla levels
        pivot = (y_high + y_low + y_close) / 3
        range_hl = y_high - y_low
        
        # Camarilla R1 and S1 (primary intraday support/resistance)
        r1 = y_close + range_hl * 1.1 / 12
        s1 = y_close - range_hl * 1.1 / 12
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_200_4h = ema_200_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above R1 AND above 4h EMA200 (uptrend)
                if curr_close > r1 and curr_close > curr_ema_200_4h:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below S1 AND below 4h EMA200 (downtrend)
                elif curr_close < s1 and curr_close < curr_ema_200_4h:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S1 or below 4h EMA200
            if curr_close < s1 or curr_close < curr_ema_200_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 or above 4h EMA200
            if curr_close > r1 or curr_close > curr_ema_200_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals