#!/usr/bin/env python3
"""
1d Camarilla Pivot R3/S3 Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Daily Camarilla R3/S3 levels act as strong intraday resistance/support.
A breakout above R3 (bullish) or below S3 (bearish) with weekly uptrend/downtrend
(EMA34) and volume spike (2x 20-day average) captures momentum moves.
Uses 1d timeframe with 1w HTF for trend. Targets 30-100 total trades over 4 years (7-25/year).
Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-day volume MA for daily volume confirmation
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for daily data and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Need prior day's OHLC for Camarilla calculation
        if i == 0:
            signals[i] = 0.0
            continue
            
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_ma_20 = vol_ma_20_1d[i]
        
        # Calculate Camarilla levels for today based on yesterday's OHLC
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla R3 and S3 levels
        r3 = prev_close + range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 4
        
        # Volume confirmation: current daily volume > 2.0 * 20-day average
        volume_confirm = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Look for entry signals
            # Long: Close above R3 AND price > weekly EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_close > r3 and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Close below S3 AND price < weekly EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_close < s3 and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Close below yesterday's close (mean reversion) OR weekly trend turns down
            if (curr_close < prev_close or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Close above yesterday's close (mean reversion) OR weekly trend turns up
            if (curr_close > prev_close or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0