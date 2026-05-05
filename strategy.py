#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout + Volume Spike + 4h Trend Filter
# Long when price breaks above Camarilla R3 (1d) with volume > 1.5x 20-bar avg AND 4h close > 4h EMA50
# Short when price breaks below Camarilla S3 (1d) with volume > 1.5x 20-bar avg AND 4h close < 4h EMA50
# Exit when price returns to Camarilla pivot point (PP) or opposite S/R level is touched
# Uses 1d for pivot levels (structure), 4h for trend filter, 1h for entry timing and volume confirmation
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.20
# Session filter: 08-20 UTC to avoid low-volume Asian session noise
# Works in bull (breakouts continuation) and bear (breakdown continuation) markets

name = "1h_Camarilla_Breakout_Volume_4hTrend"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for reliable pivots
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    # Camarilla equations: Range = (high - low)
    # R4 = close + Range * 1.1/2
    # R3 = close + Range * 1.1/4
    # R2 = close + Range * 1.1/6
    # R1 = close + Range * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - Range * 1.1/12
    # S2 = close - Range * 1.1/6
    # S3 = close - Range * 1.1/4
    # S4 = close - Range * 1.1/2
    
    daily_range = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + daily_range * 1.1 / 4.0
    camarilla_s3 = close_1d - daily_range * 1.1 / 4.0
    camarilla_r4 = close_1d + daily_range * 1.1 / 2.0
    camarilla_s4 = close_1d - daily_range * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 1h
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h volume average (20-bar) for volume spike filter
    volume_series = pd.Series(volume)
    vol_avg_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = vol_avg_20 * 1.5  # Volume > 1.5x 20-bar average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(volume_spike_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike AND 4h trend up (close > EMA50)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume[i] > volume_spike_threshold[i] and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3 with volume spike AND 4h trend down (close < EMA50)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume[i] > volume_spike_threshold[i] and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price returns to PP or breaks below S1 (opposite side)
            if close[i] <= camarilla_pp_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price returns to PP or breaks above R1 (opposite side)
            if close[i] >= camarilla_pp_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals