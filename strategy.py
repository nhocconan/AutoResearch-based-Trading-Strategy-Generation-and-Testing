#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d ADX and 6h Bollinger Bands squeeze
# ADX > 25 indicates trending market (works in bull/bear), Bollinger Band width < 50th percentile indicates low volatility/squeeze
# Enter long when price breaks above upper BB in trending regime, short when breaks below lower BB
# Exit when BB width expands above 70th percentile (end of squeeze) or opposite signal
# This captures breakouts from low volatility periods in trending markets, which works in both bull and bear regimes
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ADX_BB_Squeeze_Breakout_v1"
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
    
    # Get daily data for ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX (14-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands (20, 2) on 6h data
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        bb_width_pct = bb_width_percentile[i]
        
        # Trending market filter: ADX > 25
        is_trending = adx_val > 25
        
        # Low volatility squeeze: BB width below 50th percentile
        is_squeeze = bb_width_pct < 0.5
        
        # High volatility expansion: BB width above 70th percentile (exit condition)
        is_expansion = bb_width_pct > 0.7
        
        if position == 0:
            # Enter long: break above upper BB in trending squeeze
            if price > bb_upper[i] and is_trending and is_squeeze:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower BB in trending squeeze
            elif price < bb_lower[i] and is_trending and is_squeeze:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: BB expansion or price below lower BB
            if is_expansion or price < bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: BB expansion or price above upper BB
            if is_expansion or price > bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals