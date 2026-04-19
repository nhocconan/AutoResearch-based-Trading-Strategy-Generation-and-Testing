#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_ChoppyBreakout_Volume_Squeeze_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d_series = pd.Series(close_1d)
    bb_mid = close_1d_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (252-day lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB Width in lowest 10% percentile
    squeeze = bb_width_percentile <= 10
    
    # Align squeeze to 4h
    squeeze_4h = align_htf_to_ltf(prices, df_1d, squeeze)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h_series = pd.Series(high)
    low_4h_series = pd.Series(low)
    donchian_high = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 252)  # Ensure we have enough data for BB percentile
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(squeeze_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above Donchian high during BB squeeze with volume
            if price > donchian_high[i] and squeeze_4h[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low during BB squeeze with volume
            elif price < donchian_low[i] and squeeze_4h[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to Donchian low (mean reversion)
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to Donchian high (mean reversion)
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals