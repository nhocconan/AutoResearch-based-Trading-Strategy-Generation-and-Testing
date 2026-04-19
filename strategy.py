#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_DonchianBreakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for volatility filter and stop loss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # Fix first value
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h Donchian channels (20 periods)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(donchian_upper[i]) or \
           np.isnan(donchian_lower[i]) or np.isnan(ema50[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter
        uptrend = price > ema50[i]
        downtrend = price < ema50[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + uptrend + volume
            if price > donchian_upper[i] and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + downtrend + volume
            elif price < donchian_lower[i] and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian lower or trend reverses
            if price < donchian_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian upper or trend reverses
            if price > donchian_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals