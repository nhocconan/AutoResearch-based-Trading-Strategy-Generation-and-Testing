#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d trend filter
# Long: Price breaks above 6h Donchian upper (20) + 12h volume > 1.5x 20-period avg + 1d close > 1d EMA50
# Short: Price breaks below 6h Donchian lower (20) + 12h volume > 1.5x 20-period avg + 1d close < 1d EMA50
# Exit: Opposite Donchian break (short exit on upper break, long exit on lower break)
# Designed to capture sustained moves in both bull and bear markets with volume confirmation
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(vol_ma_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 6s Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        donchian_upper = np.max(high_window) if len(high_window) > 0 else 0
        donchian_lower = np.min(low_window) if len(low_window) > 0 else 0
        
        price = prices['close'].iloc[i]
        vol_12h = volume_12h[i // 4] if i // 4 < len(volume_12h) else 0  # Approximate 12h volume index
        
        if position == 0:
            # Long: Donchian breakout up + volume confirmation + 1d uptrend
            if (price > donchian_upper and 
                vol_12h > 1.5 * vol_ma_12h_aligned[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down + volume confirmation + 1d downtrend
            elif (price < donchian_lower and 
                  vol_12h > 1.5 * vol_ma_12h_aligned[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakdown (price < lower band)
            if price < donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout (price > upper band)
            if price > donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Volume12h_Trend1d"
timeframe = "6h"
leverage = 1.0