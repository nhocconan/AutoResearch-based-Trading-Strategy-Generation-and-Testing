#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1d choppiness regime filter.
# Enter long when price breaks above Camarilla R3 level with 1d CHOP < 38.2 (trending) and volume > 2x 24-bar average.
# Enter short when price breaks below Camarilla S3 level with 1d CHOP < 38.2 and volume confirmation.
# Exit when price retraces to Camarilla H3/L3 levels respectively.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# 1d CHOP filter ensures we only trade in trending markets, avoiding choppy conditions where breakouts fail.
# Volume spike confirms strong institutional participation. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1dCHOP_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, CHOP, and volume
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 12h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels (R3/S3 for breakouts, H3/L3 for exits)
    R3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    S3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    H3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    L3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    
    # 1d Choppiness Index (CHOP) - higher = more choppy, lower = more trending
    # CHOP = 100 * log10(sum(ATR14) / (log10(n) * (max(high_n) - min(low_n))))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average (skip NaN)
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Choppiness Index calculation
    atr_sum = np.full_like(atr_1d, np.nan)
    for i in range(14, len(atr_1d)):
        atr_sum[i] = np.nansum(atr_1d[i-13:i+1])  # Sum of last 14 ATR values
    
    # Highest high and lowest low over 14 periods
    max_high = np.full_like(high_1d, np.nan)
    min_low = np.full_like(low_1d, np.nan)
    for i in range(14, len(high_1d)):
        max_high[i] = np.nanmax(high_1d[i-13:i+1])
        min_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    # CHOP = 100 * log10(atr_sum / (log10(14) * (max_high - min_low)))
    log10_14 = np.log10(14)
    denominator = log10_14 * (max_high - min_low)
    chop_1d = np.full_like(atr_1d, np.nan)
    mask = (denominator > 0) & (~np.isnan(denominator)) & (~np.isnan(atr_sum))
    chop_1d[mask] = 100 * np.log10(atr_sum[mask] / denominator[mask])
    
    # Align CHOP to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: >2x 24-bar average volume (strict to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(volume_ma_24[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(H3[i]) or np.isnan(L3[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d CHOP trend: CHOP < 38.2 indicates trending market (not choppy)
        chop_trending = chop_aligned[i] < 38.2
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, CHOP trending, volume confirm
            if price > R3[i] and chop_trending and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3, CHOP trending, volume confirm
            elif price < S3[i] and chop_trending and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H3
            if price <= H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at L3
            if price >= L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals