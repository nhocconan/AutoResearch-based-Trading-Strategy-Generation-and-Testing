#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 1d HTF for EMA34 to capture long-term trend and reduce false breakouts in bear markets.
# Camarilla R3/S3 levels from prior completed 4h bar provide precise intraday support/resistance.
# Volume confirmation at 2.5x average ensures strong participation while limiting trades (~20-50/year).
# Session filter (08-20 UTC) reduces noise during low-liquidity periods.
# Discrete sizing 0.25 minimizes fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from prior completed 4h bar (shift by 1)
    if len(prices) < 2:
        return np.zeros(n)
    
    # Get prior completed 4h bar's OHLC (shift by 1 for 4h timeframe)
    prev_high_4h = prices['high'].shift(1).values
    prev_low_4h = prices['low'].shift(1).values
    prev_close_4h = prices['close'].shift(1).values
    
    # Camarilla pivot calculation for prior 4h bar
    pivot = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla R3 and S3 levels
    r3 = pivot + (range_4h * 1.1 / 4.0)  # R3 = pivot + (range * 1.1/4)
    s3 = pivot - (range_4h * 1.1 / 4.0)  # S3 = pivot - (range * 1.1/4)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.5x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND price > 1d EMA34 AND volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND price < 1d EMA34 AND volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below S3 OR price < 1d EMA34
            if close[i] < s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above R3 OR price > 1d EMA34
            if close[i] > r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals