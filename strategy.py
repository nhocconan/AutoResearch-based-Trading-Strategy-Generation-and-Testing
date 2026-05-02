#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation
# Uses 12h Camarilla pivot levels (R4/S4) for structure-based breakouts with institutional validation
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend whipsaws in bear markets
# Volume spike (>2.0 * 20-period EMA) confirms strong participation
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# Uses actual 12h Camarilla calculations (not resampled) for structure

name = "12h_Camarilla_R4S4_1dEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h data for Camarilla pivots (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (R4, S4)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formula: Pivot = (H + L + C) / 3
    # R4 = Pivot + (H - L) * 1.1
    # S4 = Pivot - (H - L) * 1.1
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r4_12h = pivot_12h + (high_12h - low_12h) * 1.1
    s4_12h = pivot_12h - (high_12h - low_12h) * 1.1
    
    # Align Camarilla levels to 12h timeframe (completed 12h bar only)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (strict filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for EMA50 and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above R4 with volume spike
                if close[i] > r4_aligned[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below S4 with volume spike
                if close[i] < s4_aligned[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below S4 or price below 1d EMA50
            if close[i] < s4_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R4 or price above 1d EMA50
            if close[i] > r4_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals