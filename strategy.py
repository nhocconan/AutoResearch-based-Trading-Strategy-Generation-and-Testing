# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day volume profile high-volume nodes (HVN) with 1-week trend filter.
# HVN act as dynamic support/resistance where price tends to consolidate or reverse.
# Long when price approaches HVN from below in uptrend with volume confirmation.
# Short when price approaches HVN from above in downtrend with volume confirmation.
# Uses 1-week ADX > 25 to filter for trending markets only, avoiding chop.
# Designed for low trade frequency (15-25/year) to minimize false breakouts in ranges.

name = "6h_VolumeProfile_HVN_TrendFilter"
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
    
    # Get daily data for volume profile calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate volume profile: price bins and volume distribution
    # Use 20 price bins between daily low and high
    hvn = np.zeros_like(close_1d)  # High Volume Node (price level with max volume)
    
    for i in range(len(close_1d)):
        if i < 20:  # Need minimum lookback for stable calculation
            hvn[i] = np.nan
            continue
            
        # Lookback period for volume profile (20 days)
        lookback = 20
        start_idx = max(0, i - lookback + 1)
        
        # Price range in lookback period
        period_high = np.max(high_1d[start_idx:i+1])
        period_low = np.min(low_1d[start_idx:i+1])
        
        if period_high <= period_low:
            hvn[i] = np.nan
            continue
            
        # Create price bins
        n_bins = 20
        bin_edges = np.linspace(period_low, period_high, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Volume distribution across bins
        bin_volumes = np.zeros(n_bins)
        
        for j in range(start_idx, i+1):
            # Find which bin this day's typical price falls into
            typical_price = (high_1d[j] + low_1d[j] + close_1d[j]) / 3
            bin_idx = np.searchsorted(bin_edges, typical_price) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))
            bin_volumes[bin_idx] += volume_1d[j]
        
        # Find bin with maximum volume (HVN)
        if np.sum(bin_volumes) > 0:
            max_vol_idx = np.argmax(bin_volumes)
            hvn[i] = bin_centers[max_vol_idx]
        else:
            hvn[i] = np.nan
    
    # First 20 days have no data
    hvn[:20] = np.nan
    
    # Align HVN to 6h timeframe
    hvn_aligned = align_htf_to_ltf(prices, df_1d, hvn)
    
    # Get weekly trend filter using ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smooth TR and DM
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(arr[1:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        
        # DX and ADX
        dx = np.full_like(atr, np.nan)
        mask = (plus_di + minus_di) > 0
        dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
        
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_14 = calculate_adx(high_1w, low_1w, close_1w, 14)
    # Strong trend: ADX > 25
    strong_trend = adx_14 > 25
    strong_trend = np.concatenate([[False], strong_trend[1:]])  # Align with index
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    
    # 6x EMA(50) for dynamic support/resistance and trend bias
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(hvn_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(strong_trend_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price approaching HVN from below in uptrend with volume
            if (strong_trend_aligned[i] > 0.5 and  # Strong uptrend (ADX > 25)
                close[i] > ema_50[i] and             # Above EMA50 (uptrend bias)
                close[i] <= hvn_aligned[i] * 1.01 and  # Within 1% above HVN
                close[i] >= hvn_aligned[i] * 0.99 and  # Within 1% below HVN
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price approaching HVN from above in downtrend with volume
            elif (strong_trend_aligned[i] > 0.5 and  # Strong downtrend (ADX > 25)
                  close[i] < ema_50[i] and           # Below EMA50 (downtrend bias)
                  close[i] >= hvn_aligned[i] * 0.99 and  # Within 1% below HVN
                  close[i] <= hvn_aligned[i] * 1.01 and  # Within 1% above HVN
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves significantly away from HVN or trend weakens
            if close[i] < hvn_aligned[i] * 0.97 or strong_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves significantly away from HVN or trend weakens
            if close[i] > hvn_aligned[i] * 1.03 or strong_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals