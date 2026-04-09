#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# Camarilla levels (R3/S3, R4/S4) provide institutional support/resistance
# Breakout above R4 or below S4 with volume confirmation captures strong moves
# ADX(14) > 25 filters for trending markets to avoid false breakouts in ranging conditions
# Works in bull/bear: breakouts capture momentum, volume avoids traps, ADX ensures trend quality
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Pivot + (Range * 1.1/2)
    # R3 = Pivot + (Range * 1.1/4)
    # S3 = Pivot - (Range * 1.1/4)
    # S4 = Pivot - (Range * 1.1/2)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    r3_1d = pivot_1d + (range_1d * 1.1 / 4.0)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4.0)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Calculate 1d ADX(14) for trend strength
    def calculate_dmx(high, low, close):
        """Calculate +DM and -DM"""
        high_diff = np.diff(high, prepend=high[0])
        low_diff = np.diff(low, prepend=low[0])
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        return plus_dm, minus_dm
    
    def wilders_smoothing(values, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    plus_dm, minus_dm = calculate_dmx(high_1d, low_1d, close_1d)
    
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below R3 (profit taking) OR trend weakens
            if close[i] < r3_1d_aligned[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 (profit taking) OR trend weakens
            if close[i] > s3_1d_aligned[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: breakout with volume confirmation in trending market
            if trending and volume_confirmed:
                if close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals