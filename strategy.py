#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Breakout with 12h Trend Filter and Volume Confirmation
# Uses 6h Donchian channel breakout for entry, filtered by 12h ADX > 25 for trend strength
# and volume > 1.5x 20-period average for confirmation. Exits on opposite Donchian breakout
# or when trend weakens (ADX < 20). Designed to work in both bull and bear markets
# by capturing strong trending moves with volume validation.
# Target: 15-35 trades/year via Donchian + trend + volume confluence.

name = "6h_donchian_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 14-period ADX for trend strength on 12h data
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 12h values for current 6h bar
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx)[i]
        
        # Trend filter: ADX > 25 for strong trend, ADX < 20 for weak trend
        strong_trend = adx_aligned > 25
        weak_trend = adx_aligned < 20
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_high[i-1]  # Break above upper band
        short_breakout = close[i] < donchian_low[i-1]  # Break below lower band
        
        if position == 1:  # Long position
            # Exit: opposite breakout OR trend weakens
            if short_breakout or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: opposite breakout OR trend weakens
            if long_breakout or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade during strong trend with volume confirmation
            if strong_trend and volume_confirm:
                # Long on upward breakout
                if long_breakout:
                    position = 1
                    signals[i] = 0.25
                # Short on downward breakout
                elif short_breakout:
                    position = -1
                    signals[i] = -0.25
    
    return signals