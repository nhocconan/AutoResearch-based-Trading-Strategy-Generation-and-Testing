#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels (R3/S3 fade, R4/S4 breakout) with 1d EMA trend filter and volume confirmation
# Long when price breaks above 1w Camarilla R4 AND 1d EMA34 > EMA89 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Camarilla S4 AND 1d EMA34 < EMA89 AND volume > 1.5 * avg_volume(20)
# Fade longs at 1w Camarilla R3 when price rejects (close < open) with volume confirmation
# Fade shorts at 1w Camarilla S3 when price rejects (close > open) with volume confirmation
# Exit when price touches 1w Camarilla midpoint (R3/S3 midpoint) or opposite Camarilla level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1w Camarilla provides strong weekly structural levels
# 1d EMA filter ensures alignment with daily trend, reducing counter-trend trades
# Volume confirmation filters weak breakouts/fades
# Works in bull (breakout continuation) and bear (fade at weekly resistance/support)

name = "6h_1wCamarilla_R3S3_R4S4_1dEMATrend_Volume"
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
    open_price = prices['open'].values
    
    # Get 1w data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 completed 1w bars for pivot calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels using previous week's OHLC
    # Camarilla: based on previous period's range
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan  # First value has no previous
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels
    r3_1w = pivot_1w + (range_1w * 1.1 / 4)
    s3_1w = pivot_1w - (range_1w * 1.1 / 4)
    r4_1w = pivot_1w + (range_1w * 1.1 / 2)
    s4_1w = pivot_1w - (range_1w * 1.1 / 2)
    m3_1w = (r3_1w + s3_1w) / 2.0  # Midpoint between R3/S3
    
    # Align 1w Camarilla levels to 6h timeframe (wait for completed 1w bar)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    m3_aligned = align_htf_to_ltf(prices, df_1w, m3_1w)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA89
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMA values to 6h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(m3_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 1w Camarilla R4 with 1d EMA34 > EMA89 and volume confirmation
            if (close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 1w Camarilla S4 with 1d EMA34 < EMA89 and volume confirmation
            elif (close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            # Long fade: price rejects at 1w Camarilla R3 (close < open) with volume confirmation
            elif (close[i] < r3_aligned[i] and close[i-1] > r3_aligned[i-1] and 
                  close[i] < open_price[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short fade: price rejects at 1w Camarilla S3 (close > open) with volume confirmation
            elif (close[i] > s3_aligned[i] and close[i-1] < s3_aligned[i-1] and 
                  close[i] > open_price[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1w Camarilla midpoint or S3 (reversal or profit take)
            if close[i] <= m3_aligned[i] or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1w Camarilla midpoint or R3 (reversal or profit take)
            if close[i] >= m3_aligned[i] or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals