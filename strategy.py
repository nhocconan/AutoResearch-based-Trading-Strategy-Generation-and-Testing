#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation.
# Long when price breaks above upper BB after squeeze (BB width < 50th percentile) AND 1d volume > 1.5x 20-day average.
# Short when price breaks below lower BB after squeeze with same volume condition.
# Exit when price returns to middle band.
# Bollinger squeeze identifies low volatility breakouts, volume confirms institutional interest.
# Works in both bull/bear markets by capturing volatility expansion phases.
# Target: 15-25 trades/year per symbol (90-100 total over 4 years).
name = "6h_BollingerSqueeze_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d average volume (20-day)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma + 2 * std
    lower_bb = ma - 2 * std
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (50-period lookback) for squeeze condition
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure BB and percentile are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ma[i]) or np.isnan(std[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        width_percentile = bb_width_percentile[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        mid = ma[i]
        
        # Squeeze condition: BB width below 50th percentile (low volatility)
        squeeze = width_percentile < 50
        
        if position == 0:
            # Long entry: break above upper BB after squeeze + volume spike
            if price > upper and squeeze and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower BB after squeeze + volume spike
            elif price < lower and squeeze and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band
            if price >= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band
            if price <= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals