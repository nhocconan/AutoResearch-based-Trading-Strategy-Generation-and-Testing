#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Uses price channel breakouts (proven structure) with ADX>25 to ensure trending markets
# Volume confirmation filters false breakouts. Works in bull/bear: ADX adapts to trend strength
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Donchian20_ADX_VolumeSpike_1dTrend_v1"
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
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['low']).shift(1)
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']) - pd.Series(df_1d['high']).shift(1)
    dm_minus = pd.Series(df_1d['low']).shift(1) - pd.Series(df_1d['low'])
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    tr_14 = tr.rolling(window=14, min_periods=14).sum()
    dm_plus_14 = dm_plus.rolling(window=14, min_periods=14).sum()
    dm_minus_14 = dm_minus.rolling(window=14, min_periods=14).sum()
    
    # DI and ADX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_adx = adx_aligned[i]
        
        # Only trade in trending markets (ADX > 25) with volume confirmation
        if curr_adx > 25 and curr_volume_confirm:
            if position == 0:  # Flat - look for new entries
                # Bullish breakout: price breaks above Donchian high
                if curr_high > curr_donch_high:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Donchian low
                elif curr_low < curr_donch_low:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:  # Long position
                # Exit when price breaks below Donchian low (reversal)
                if curr_low < curr_donch_low:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:  # Short position
                # Exit when price breaks above Donchian high (reversal)
                if curr_high > curr_donch_high:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals