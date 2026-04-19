#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyBreakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 400:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w - low_1w, np.absolute(high_1w - np.roll(close_1w, 1)), np.absolute(low_1w - np.roll(close_1w, 1)))
    tr_1w[0] = high_1w[0] - low_1w[0]  # Fix first value
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Weekly Donchian channels for breakout signals
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or \
           np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend bias: long bias if price > weekly EMA200, short bias if price < weekly EMA200
        long_bias = price > ema200_1w_aligned[i]
        short_bias = price < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: breakout above weekly Donchian high + above weekly EMA200 + volume
            if price > donchian_high_20_aligned[i] and long_bias and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly Donchian low + below weekly EMA200 + volume
            elif price < donchian_low_20_aligned[i] and short_bias and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price drops below weekly EMA200 or breaks below weekly Donchian low
            if price < ema200_1w_aligned[i] or price < donchian_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above weekly EMA200 or breaks above weekly Donchian high
            if price > ema200_1w_aligned[i] or price > donchian_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals