#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R3/S3, R4/S4) with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above weekly R4 AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 6h
# Short when price breaks below weekly S4 AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 6h
# Exit when price retests weekly R3 (for longs) or S3 (for shorts)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla provides strong structural levels with institutional relevance
# 1d EMA34 ensures we trade with the dominant daily trend filter
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "6h_WeeklyCamarilla_R4S4_1dEMA34_Volume"
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
    
    # Get weekly data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least 1 completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    #          S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # Use previous week's values to avoid look-ahead
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan  # First value has no previous
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    weekly_range = prev_high_1w - prev_low_1w
    r4 = prev_close_1w + (weekly_range * 1.1 / 2)
    r3 = prev_close_1w + (weekly_range * 1.1 / 4)
    s3 = prev_close_1w - (weekly_range * 1.1 / 4)
    s4 = prev_close_1w - (weekly_range * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed weekly bar)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R4, 1d EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (close[i] > r4_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4, 1d EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (close[i] < s4_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests weekly R3
            if close[i] <= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests weekly S3
            if close[i] >= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals