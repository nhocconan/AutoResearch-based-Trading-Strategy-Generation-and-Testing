#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R3/S3) for mean reversion entries
# with 1d EMA34 trend filter and volume spike confirmation
# Long when price touches or breaks below S3 AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 2.0 * avg_volume(20)
# Short when price touches or breaks above R3 AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 2.0 * avg_volume(20)
# Exit when price crosses the 1d VWAP (mean reversion to fair value)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla levels provide strong support/resistance from institutional order flow
# 1d EMA34 trend filter ensures we trade with the dominant daily trend to avoid counter-trend whipsaws
# Volume spike confirmation validates institutional participation at key levels
# Works in both bull (buy dips at S3 in uptrend) and bear (sell rallies at R3 in downtrend) markets

name = "6h_1wCamarilla_R3S3_MeanReversion_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least 1 completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1
    s3_1w = pivot_1w - range_1w * 1.1
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed 1w bar)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Get 1d data ONCE before loop for EMA34 and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d VWAP (volume-weighted average price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d = np.where(volume_1d.cumsum() == 0, typical_price_1d, vwap_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
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
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at or below S3, 1d EMA34 uptrend, volume spike, in session
            if (close[i] <= s3_1w_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price at or above R3, 1d EMA34 downtrend, volume spike, in session
            elif (close[i] >= r3_1w_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above 1d VWAP (mean reversion)
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below 1d VWAP (mean reversion)
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals