#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Camarilla pivot levels with 6h trend filter and volume confirmation
# Long when price breaks above Camarilla R3 (1d) AND 6h close > 6h EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S3 (1d) AND 6h close < 6h EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price touches Camarilla pivot point (PP) or opposite Camarilla level (R4/S4 for continuation)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Daily Camarilla provides strong intraday support/resistance levels
# 6h EMA50 filter ensures alignment with intermediate-term trend, reducing counter-trend trades
# High volume threshold (2.0x) filters weak breakouts and ensures institutional participation
# Works in bull (breakout continuation at R4/S4) and bear (mean reversion at R3/S3)

name = "6h_1dCamarilla_R3S3_6hEMA50_Volume"
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
    
    # Get 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation (requires prior day's OHLC)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # PP = (Prior Day High + Prior Day Low + Prior Day Close) / 3
    # R1 = PP + (Prior Day High - Prior Day Low) * 1.1/12
    # R2 = PP + (Prior Day High - Prior Day Low) * 1.1/6
    # R3 = PP + (Prior Day High - Prior Day Low) * 1.1/4
    # R4 = PP + (Prior Day High - Prior Day Low) * 1.1/2
    # S1 = PP - (Prior Day High - Prior Day Low) * 1.1/12
    # S2 = PP - (Prior Day High - Prior Day Low) * 1.1/6
    # S3 = PP - (Prior Day High - Prior Day Low) * 1.1/4
    # S4 = PP - (Prior Day High - Prior Day Low) * 1.1/2
    pp_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3.0
    range_1d = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    r3_1d = pp_1d + range_1d * 1.1 / 4.0
    s3_1d = pp_1d - range_1d * 1.1 / 4.0
    r4_1d = pp_1d + range_1d * 1.1 / 2.0
    s4_1d = pp_1d - range_1d * 1.1 / 2.0
    pp_1d_aligned = pp_1d  # will be aligned below
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Get 6h data ONCE before loop for EMA trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA50
    close_series_6h = pd.Series(close_6h)
    ema_50_6h = close_series_6h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h EMA50 to 6h timeframe (no additional delay needed for EMA)
    ema_50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_50_6h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_50_6h_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 6h EMA50 uptrend and volume confirmation
            if (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1] and 
                close[i] > ema_50_6h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 6h EMA50 downtrend and volume confirmation
            elif (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1] and 
                  close[i] < ema_50_6h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Camarilla PP (mean reversion) or R4 (continuation exhaustion)
            if close[i] <= pp_1d_aligned[i] or close[i] >= r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1d Camarilla PP (mean reversion) or S4 (continuation exhaustion)
            if close[i] >= pp_1d_aligned[i] or close[i] <= s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals