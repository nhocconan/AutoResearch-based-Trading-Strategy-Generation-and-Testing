#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + chop regime filter
# Camarilla pivots from 1d provide precise intraday support/resistance levels
# Volume confirmation ensures breakout authenticity (avoids fakeouts)
# Choppiness index regime filter: CHOP > 61.8 = range (mean revert at levels), CHOP < 38.2 = trending (follow breakout)
# Works in bull/bear: regime filter adapts, Camarilla levels provide structure in all markets
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla, volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d typical price for Camarilla
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_1d = typical_price_1d.values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # H5 = TP + 1.1*(H-L)/2, H4 = TP + 1.1*(H-L)/4, H3 = TP + 1.1*(H-L)/6
    # L3 = TP - 1.1*(H-L)/6, L4 = TP - 1.1*(H-L)/4, L5 = TP - 1.1*(H-L)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate levels using previous day's data (shifted by 1 to avoid look-ahead)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_tp = (prev_high + prev_low + prev_close) / 3
    
    # Camarilla levels
    H5 = prev_tp + 1.1 * (prev_high - prev_low) / 2
    H4 = prev_tp + 1.1 * (prev_high - prev_low) / 4
    H3 = prev_tp + 1.1 * (prev_high - prev_low) / 6
    L3 = prev_tp - 1.1 * (prev_high - prev_low) / 6
    L4 = prev_tp - 1.1 * (prev_high - prev_low) / 4
    L5 = prev_tp - 1.1 * (prev_high - prev_low) / 2
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    H5_aligned = align_htf_to_ltf(prices, df_1d, H5)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    L5_aligned = align_htf_to_ltf(prices, df_1d, L5)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H5_aligned[i]) or np.isnan(L5_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 1d average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 40 = strong trend, CHOP > 60 = strong range, middle = neutral
        strong_trend = chop_1d_aligned[i] < 40
        strong_range = chop_1d_aligned[i] > 60
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (Camarilla support) OR strong ranging regime
            if close[i] < L3_aligned[i] or strong_range:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (Camarilla resistance) OR strong ranging regime
            if close[i] > H3_aligned[i] or strong_range:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if strong_trend:
                # Follow breakout in strong trending regime
                if close[i] > H4_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < L4_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif strong_range:
                # Mean revert at Camarilla H3/L3 levels in strong ranging regime
                if close[i] < L3_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > H3_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            else:  # Neutral regime
                # Moderate breakout follow or mean revert at H4/L4
                if close[i] > H4_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < L4_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals