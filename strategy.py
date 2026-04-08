#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX trend strength filter combined with Bollinger Band squeeze breakout
# Uses ADX to identify trending regimes (>25) and Bollinger Band width percentile to detect squeeze
# Breakouts from Bollinger Bands with ADX confirmation capture momentum moves in both bull and bear markets
# Bollinger Band squeeze acts as a volatility filter, reducing false breakouts in choppy markets
# Target: 12-37 trades/year by requiring both volatility contraction and trend confirmation
name = "12h_adx_bb_squeeze_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend strength
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
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band width percentile (50-period) to identify squeeze
    bb_width_pct = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(bb_width_pct[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 12h bar
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)[i]
        bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)[i]
        upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)[i]
        lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)[i]
        sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned > 25
        
        # Squeeze filter: BB width below 20th percentile indicates volatility contraction
        squeeze = bb_width_pct_aligned < 0.2
        
        if position == 1:  # Long position
            # Exit: price closes below middle Bollinger Band OR trend weakens
            if close[i] < sma_20_aligned or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle Bollinger Band OR trend weakens
            if close[i] > sma_20_aligned or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade during volatility squeeze with strong trend
            if squeeze and strong_trend:
                # Long: price breaks above upper Bollinger Band
                if close[i] > upper_bb_aligned:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Bollinger Band
                elif close[i] < lower_bb_aligned:
                    position = -1
                    signals[i] = -0.25
    
    return signals