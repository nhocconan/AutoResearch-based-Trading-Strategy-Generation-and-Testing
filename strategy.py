#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_ChoppyBreakout_Volume_Squeeze_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20-period, 2.0 std)
    close_1w_series = pd.Series(close_1w)
    basis_1w = close_1w_series.rolling(window=20, min_periods=20).mean().values
    dev_1w = close_1w_series.rolling(window=20, min_periods=20).std().values
    upper_1w = basis_1w + 2.0 * dev_1w
    lower_1w = basis_1w - 2.0 * dev_1w
    
    # Bandwidth = (upper - lower) / basis
    bw_1w = (upper_1w - lower_1w) / basis_1w
    # Percentile rank of bandwidth (50-period)
    bw_percentile = pd.Series(bw_1w).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Choppiness condition: low volatility regime (bandwidth < 20th percentile)
    choppy = bw_percentile < 0.20
    
    # Align choppy signal to 4h
    choppy_4h = align_htf_to_ltf(prices, df_1w, choppy)
    
    # 4h Donchian channel (20-period)
    high_4h_series = pd.Series(high)
    low_4h_series = pd.Series(low)
    donchian_high = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(choppy_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        is_choppy = choppy_4h[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Break above Donchian high in choppy market with volume
            if price > donchian_high[i] and is_choppy and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in choppy market with volume
            elif price < donchian_low[i] and is_choppy and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to midline (mean reversion in chop)
            midline = (donchian_high[i] + donchian_low[i]) / 2.0
            if price < midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to midline
            midline = (donchian_high[i] + donchian_low[i]) / 2.0
            if price > midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals