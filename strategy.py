# 6h_camarilla_pivot_1d_trend_volume_v1
# Hypothesis: Camarilla pivot levels from daily data provide high-probability reversal (R3/S3) and breakout (R4/S4) levels. 
# Combined with volume confirmation and trend filter from daily ADX, this captures institutional interest at key levels.
# Designed for 15-30 trades/year per symbol to minimize fee drag and work in both bull/bear markets.

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.0 / 12)
    r2 = pivot + (range_hl * 2.0 / 12)
    r3 = pivot + (range_hl * 3.0 / 12)
    r4 = pivot + (range_hl * 1.5 / 6)  # 3/12 = 1.5/6
    
    # Support levels
    s1 = pivot - (range_hl * 1.0 / 12)
    s2 = pivot - (range_hl * 2.0 / 12)
    s3 = pivot - (range_hl * 3.0 / 12)
    s4 = pivot - (range_hl * 1.5 / 6)  # 3/12 = 1.5/6
    
    # Calculate 14-period ADX for trend strength on 1d data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values with proper min_periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40  # Need ADX and volume buffers
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in strong trending markets
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (reversal zone)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (reversal zone)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation and in strong trending markets
            if volume_confirm and strong_trend:
                # Long entry: price breaks above R4 (breakout continuation)
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below S4 (breakout continuation)
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                # Long reversal: price bounces from S3 (support zone)
                elif close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short reversal: price rejects from R3 (resistance zone)
                elif close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals