#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w ADX filter.
In trending markets (ADX>25), buy breakouts above upper band, sell breakdowns below lower band.
In ranging markets (ADX<20), fade extremes with mean reversion to the middle.
Position sizing: 0.30 for entries, 0 for exits.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period high/low)
    upper_dc = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_dc = (upper_dc + lower_dc) / 2
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1d
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, lower_dc)
    middle_dc_aligned = align_htf_to_ltf(prices, df_1d, middle_dc)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or 
            np.isnan(middle_dc_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Determine market regime
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:
            # Trending market: breakout strategy
            if trending:
                # Long: break above upper Donchian with volume
                if (close[i] > upper_dc_aligned[i] and 
                    volume[i] > vol_ma_20_aligned[i] * 1.5):
                    signals[i] = 0.30
                    position = 1
                # Short: break below lower Donchian with volume
                elif (close[i] < lower_dc_aligned[i] and 
                      volume[i] > vol_ma_20_aligned[i] * 1.5):
                    signals[i] = -0.30
                    position = -1
            # Ranging market: mean reversion to middle
            elif ranging:
                # Long: price near lower band, revert to middle
                if close[i] <= lower_dc_aligned[i] * 1.005:  # within 0.5% of lower band
                    signals[i] = 0.30
                    position = 1
                # Short: price near upper band, revert to middle
                elif close[i] >= upper_dc_aligned[i] * 0.995:  # within 0.5% of upper band
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit long: price reaches middle or opposite band
            if trending:
                # In trending market, hold until opposite band or momentum fails
                if close[i] >= upper_dc_aligned[i] * 1.02 or close[i] <= middle_dc_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:
                # In ranging market, exit at middle
                if close[i] >= middle_dc_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price reaches middle or opposite band
            if trending:
                # In trending market, hold until opposite band or momentum fails
                if close[i] <= lower_dc_aligned[i] * 0.98 or close[i] >= middle_dc_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
            else:
                # In ranging market, exit at middle
                if close[i] <= middle_dc_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_ADX_Volume_Regime"
timeframe = "1d"
leverage = 1.0