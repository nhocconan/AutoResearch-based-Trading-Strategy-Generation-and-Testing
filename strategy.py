#!/usr/bin/env python3
# 4h_adx_donchian_volume_v1
# Hypothesis: Breakout strategy using Donchian channels with ADX trend filter and volume confirmation.
# Works in bull markets: buy when price breaks above Donchian(20) high with rising ADX and volume surge.
# Works in bear markets: sell when price breaks below Donchian(20) low with rising ADX and volume surge.
# Uses ADX to filter weak breakouts and volume to confirm conviction.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adx_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX (14-period) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    plus_dm_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - low_1d[:-1]), np.absolute(low_1d[1:] - high_1d[:-1]))
    plus_dm_1d = np.concatenate([[0], plus_dm_1d])
    minus_dm_1d = np.concatenate([[0], minus_dm_1d])
    tr_1d = np.concatenate([[0], tr_1d])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).mean().values / atr_14_1d
    minus_di_1d = 100 * pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).mean().values / atr_14_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or ADX weakens
            if close[i] < donch_low[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or ADX weakens
            if close[i] > donch_high[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume surge and strong ADX
            if (close[i] > donch_high[i] and vol_surge and 
                adx[i] > 25 and adx_1d_aligned[i] > 20):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume surge and strong ADX
            elif (close[i] < donch_low[i] and vol_surge and 
                  adx[i] > 25 and adx_1d_aligned[i] > 20):
                position = -1
                signals[i] = -0.25
    
    return signals