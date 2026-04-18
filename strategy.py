#!/usr/bin/env python3
"""
6h_BollingerBandWidth_Squeeze_Breakout
Hypothesis: Bollinger Band Width (BBW) squeeze on 1-day timeframe precedes volatility expansion.
Breakouts from 60-period high/low on 6h chart after BBW squeeze capture explosive moves.
Works in both bull and bear markets as volatility expansion occurs in all regimes.
Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Bollinger Band Width calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Bollinger Bands: 20-period SMA ± 2 standard deviations
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Bollinger Band Width: (Upper - Lower) / Middle
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # BBW squeeze: lowest 10% of BBW over last 50 days indicates compression
    bb_width_rank = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 50 else np.nan, raw=False
    ).values
    squeeze_signal = bb_width_rank <= 0.1  # Bottom 10% = squeeze
    
    # Align squeeze signal to 6h timeframe
    squeeze_6h = align_htf_to_ltf(prices, df_1d, squeeze_signal)
    
    # 60-period high/low for breakout levels on 6h
    high_60 = pd.Series(high).rolling(window=60, min_periods=60).max().values
    low_60 = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Wait for 60-period high/low calculation
    
    for i in range(start_idx, n):
        if (np.isnan(squeeze_6h[i]) or np.isnan(high_60[i]) or 
            np.isnan(low_60[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        squeeze_active = squeeze_6h[i]
        vol_ok = volume_filter[i]
        hi_60 = high_60[i]
        lo_60 = low_60[i]
        
        if position == 0:
            # Long: break above 60-period high during BBW squeeze with volume
            if price > hi_60 and squeeze_active and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below 60-period low during BBW squeeze with volume
            elif price < lo_60 and squeeze_active and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to 60-period low or volatility expands (BBW > 80th percentile)
            bb_width_current = bb_width[min(i // 4, len(bb_width)-1)] if i // 4 < len(bb_width) else bb_width[-1]
            bb_width_rank_current = pd.Series(bb_width[:min(i//4+1, len(bb_width))]).rank(pct=True).iloc[-1] if i//4 < len(bb_width) and i//4 >= 20 else 0.5
            vol_expansion = bb_width_rank_current > 0.8 if not np.isnan(bb_width_rank_current) else False
            
            if price < lo_60 or vol_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 60-period high or volatility expands
            bb_width_current = bb_width[min(i // 4, len(bb_width)-1)] if i // 4 < len(bb_width) else bb_width[-1]
            bb_width_rank_current = pd.Series(bb_width[:min(i//4+1, len(bb_width))]).rank(pct=True).iloc[-1] if i//4 < len(bb_width) and i//4 >= 20 else 0.5
            vol_expansion = bb_width_rank_current > 0.8 if not np.isnan(bb_width_rank_current) else False
            
            if price > hi_60 or vol_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerBandWidth_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0