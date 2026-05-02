#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation
# Uses 6h Camarilla pivot levels (R4/S4) for strong breakouts with institutional validation
# 1w EMA50 ensures alignment with long-term trend to avoid counter-trend whipsaws in bear markets
# Volume spike (>2.0 * 20-period EMA) confirms strong participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# Uses actual 6h Camarilla calculations (not resampled) for structure

name = "6h_Camarilla_R4S4_1wEMA50_Trend_Volume_v1"
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
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h data for Camarilla pivots (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Calculate 6h Camarilla pivot levels (R4, S4)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Camarilla formula: Pivot = (H + L + C) / 3
    # R4 = Pivot + (H - L) * 1.1
    # S4 = Pivot - (H - L) * 1.1
    pivot_6h = (high_6h + low_6h + close_6h) / 3.0
    r4_6h = pivot_6h + (high_6h - low_6h) * 1.1
    s4_6h = pivot_6h - (high_6h - low_6h) * 1.1
    
    # Align Camarilla levels to 6h timeframe (completed 6h bar only)
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4_6h)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4_6h)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
        
        # Determine trend bias from 1w EMA50
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
            # Exit: price breaks below S4 or price below 1w EMA50
            if close[i] < s4_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R4 or price above 1w EMA50
            if close[i] > r4_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals