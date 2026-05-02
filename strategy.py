#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1h Camarilla pivot levels (R1/S1) for structure-based breakouts with institutional validation
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume spike (>1.8 * 20-period EMA) confirms strong participation
# Session filter (08-20 UTC) reduces noise trades
# Designed for low trade frequency: ~20-30 trades/year per symbol with 0.20 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# Uses actual 1h Camarilla calculations (not resampled) for structure

name = "1h_Camarilla_R1S1_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
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
    
    # 4h HTF data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1h data for Camarilla pivots (primary timeframe)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 1h Camarilla pivot levels (R1, S1)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla formula: Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 4
    # S1 = Pivot - (H - L) * 1.1 / 4
    pivot_1h = (high_1h + low_1h + close_1h) / 3.0
    r1_1h = pivot_1h + (high_1h - low_1h) * 1.1 / 4.0
    s1_1h = pivot_1h - (high_1h - low_1h) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe (completed 1h bar only)
    r1_aligned = align_htf_to_ltf(prices, df_1h, r1_1h)
    s1_aligned = align_htf_to_ltf(prices, df_1h, s1_1h)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA (moderate filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for EMA50 and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above R1 with volume spike
                if close[i] > r1_aligned[i-1] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below S1 with volume spike
                if close[i] < s1_aligned[i-1] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below S1 or price below 4h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above R1 or price above 4h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals