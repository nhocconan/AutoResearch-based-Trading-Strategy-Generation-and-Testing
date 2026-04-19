#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_TrendFilter_Breakout_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d Donchian(20) breakout levels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: price relative to weekly EMA50
        price_above_ema = price > ema50_1w_aligned[i]
        price_below_ema = price < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > donchian_high_20[i] and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < donchian_low_20[i] and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below Donchian low (mean reversion)
            if price < donchian_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above Donchian high (mean reversion)
            if price > donchian_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals