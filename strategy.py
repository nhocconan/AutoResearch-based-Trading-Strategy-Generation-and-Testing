#!/usr/bin/env python3
"""
6h_market_regime_adaptive_v1
Hypothesis: Adaptive strategy that switches between trend-following in trending markets and mean-reversion in ranging markets, using weekly ADX and price position relative to weekly Bollinger Bands. In trending markets (ADX>25), trade breakouts of daily Donchian channels. In ranging markets (ADX<=25), fade extremes of daily Bollinger Bands. Uses 60-minute volume confirmation to avoid false signals. Designed for 60-120 trades/year to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_market_regime_adaptive_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly ADX for regime detection (trend strength)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    
    # Minus Directional Movement
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip first NaN
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1w = wilders_smooth(tr, 14)
    plus_di_1w = 100 * wilders_smooth(dm_plus, 14) / atr_1w
    minus_di_1w = 100 * wilders_smooth(dm_minus, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smooth(dx_1w, 14)
    
    # Weekly Bollinger Bands for ranging market detection
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + 2 * std_20_1w
    lower_bb_1w = sma_20_1w - 2 * std_20_1w
    
    # Daily Donchian Channel for breakout signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian Channel (20-period)
    upper_dc = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily Bollinger Bands for mean reversion
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Align all indicators to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, lower_dc)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Volume confirmation (60-period average)
    vol_ma = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if data not available
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(upper_bb_1w_aligned[i]) or 
            np.isnan(lower_bb_1w_aligned[i]) or np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Regime detection: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_1w_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit conditions
            if is_trending:
                # Exit trend long when price touches lower Donchian
                if close[i] <= lower_dc_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # Exit mean reversion long when price returns to mean
                if close[i] >= sma_20_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending:
                # Exit trend short when price touches upper Donchian
                if close[i] >= upper_dc_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion short when price returns to mean
                if close[i] <= sma_20_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                if is_trending:
                    # Trending market: breakout entries
                    # Long: price breaks above upper Donchian
                    if close[i] > upper_dc_aligned[i] and close[i-1] <= upper_dc_aligned[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price breaks below lower Donchian
                    elif close[i] < lower_dc_aligned[i] and close[i-1] >= lower_dc_aligned[i-1]:
                        position = -1
                        signals[i] = -0.25
                else:
                    # Ranging market: mean reversion entries
                    # Long: price touches lower Bollinger Band
                    if close[i] <= lower_bb_1d_aligned[i] and close[i-1] > lower_bb_1d_aligned[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price touches upper Bollinger Band
                    elif close[i] >= upper_bb_1d_aligned[i] and close[i-1] < upper_bb_1d_aligned[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals

# Pre-calculate daily indicators for alignment efficiency
def _calculate_daily_indicators(df_1d):
    """Pre-calculate daily indicators to avoid recomputation in alignment"""
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20-period)
    upper_dc = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Bollinger Bands (20,2)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    return upper_dc, lower_dc, upper_bb_1d, lower_bb_1d, sma_20_1d

# Note: The actual implementation above calculates these inside the loop for clarity
# In production, these would be pre-calculated for better performance