#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for 200-bar EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    # Calculate Donchian(20) on 12h high/low (use lookback of 20 periods)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Vectorized rolling max/min for Donchian channels
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200  # Need 200 for EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above 12h Donchian high with volume confirmation and price above 1d EMA200 (uptrend)
            if price > donchian_high_aligned[i] and vol > 1.5 * avg_vol[i] and price > ema200_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: breakout below 12h Donchian low with volume confirmation and price below 1d EMA200 (downtrend)
            elif price < donchian_low_aligned[i] and vol > 1.5 * avg_vol[i] and price < ema200_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes back below 12h Donchian low (mean reversion)
            if price < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes back above 12h Donchian high (mean reversion)
            if price > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_Breakout_TrendFilter"
timeframe = "4h"
leverage = 1.0