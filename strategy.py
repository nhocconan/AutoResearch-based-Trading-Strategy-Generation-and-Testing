#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX Trend Filter
# Long when: BB squeeze (bandwidth < 20th percentile) AND price breaks above upper band AND 1d ADX > 25
# Short when: BB squeeze (bandwidth < 20th percentile) AND price breaks below lower band AND 1d ADX > 25
# Exit when: BB squeeze ends (bandwidth > 50th percentile) OR opposite band break
# Uses Bollinger Bands (20,2) for squeeze detection, 1d ADX for trend confirmation.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading breakouts during low volatility (squeeze) with trend confirmation.

name = "6h_BBSqueeze_Breakout_1dADX_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus_1d[0] = 0
    dm_minus_1d[0] = 0
    
    # Smoothed TR, DM+, DM- (14-period)
    tr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).mean().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_1d = 100 * dm_plus_14_1d / tr_14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr_14_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Bollinger Bands (20,2) on 6h
    bb_period = 20
    bb_std = 2
    
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    bandwidth = (upper_band - lower_band) / sma  # Normalized bandwidth
    
    # Calculate bandwidth percentiles for squeeze detection (using 50-period lookback)
    bandwidth_pct = pd.Series(bandwidth).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 50 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 14)  # BB, bandwidth percentile, and ADX warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(bandwidth_pct[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_bw_pct = bandwidth_pct[i]
        curr_adx = adx_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: squeeze ends (bandwidth > 50th percentile) OR price breaks below lower band
            if curr_bw_pct > 0.5 or curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: squeeze ends (bandwidth > 50th percentile) OR price breaks above upper band
            if curr_bw_pct > 0.5 or curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when: squeeze (bandwidth < 20th percentile) AND price breaks above upper band AND ADX > 25
            if curr_bw_pct < 0.2 and curr_close > curr_upper and curr_adx > 25.0:
                signals[i] = 0.25
                position = 1
            # Short when: squeeze (bandwidth < 20th percentile) AND price breaks below lower band AND ADX > 25
            elif curr_bw_pct < 0.2 and curr_close < curr_lower and curr_adx > 25.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals