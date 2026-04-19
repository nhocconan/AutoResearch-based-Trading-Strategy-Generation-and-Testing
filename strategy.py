#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeSpike_RangeFilter"
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
    
    # Get daily data for regime filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ATR for range detection (14-period)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Daily price range for range detection
    daily_range = high - low
    range_ma_10 = pd.Series(daily_range).rolling(window=10, min_periods=10).mean().values
    
    # Align daily ATR and range to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    range_ma_10_aligned = align_htf_to_ltf(prices, df_1d, range_ma_10)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume spike filter (current volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(range_ma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14_1d_aligned[i]
        range_ma = range_ma_10_aligned[i]
        
        # Range filter: only trade when volatility is elevated (avoid choppy markets)
        volatile_market = atr > 1.2 * range_ma
        
        volume_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: break above Donchian high with volume spike in volatile market
            if price > donchian_high[i] and volume_spike and volatile_market:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike in volatile market
            elif price < donchian_low[i] and volume_spike and volatile_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below Donchian low or loss of volatility
            if price < donchian_low[i] or not volatile_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above Donchian high or loss of volatility
            if price > donchian_high[i] or not volatile_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals