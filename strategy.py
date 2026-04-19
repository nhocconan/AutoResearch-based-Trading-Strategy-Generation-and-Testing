#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_VolumeSpike_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 10-period ATR for Donchian width filter
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Donchian channels (20-period) using previous day's data
    # Upper = max(high of last 20 days), Lower = min(low of last 20 days)
    # Use rolling window with shift to avoid look-ahead
    high_20_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed days
    donchian_upper = np.concatenate([[np.nan], high_20_max[:-1]])
    donchian_lower = np.concatenate([[np.nan], low_20_min[:-1]])
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 4h trend filter: EMA(50) slope
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50 - np.roll(ema_50, 1)
    ema_50_slope[0] = 0
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or \
           np.isnan(ema_50_slope[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_10[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_10[i]
        
        # Volume filter - require significant volume spike
        volume_ok = vol > 2.0 * vol_ma
        
        # Trend filter: bullish when EMA slope > 0
        bullish_trend = ema_50_slope[i] > 0
        bearish_trend = ema_50_slope[i] < 0
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and bullish trend
            if price > donchian_upper_aligned[i] and volume_ok and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume spike and bearish trend
            elif price < donchian_lower_aligned[i] and volume_ok and bearish_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below Donchian upper or trend turns bearish
            if price < donchian_upper_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above Donchian lower or trend turns bullish
            if price > donchian_lower_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals