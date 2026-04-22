#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for calculations (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Daily ATR-based volatility filter
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    vol_ratio = atr_14 / atr_50
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # 12h volume confirmation (20-period average)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready or outside session
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(vol_avg_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume AND high volatility regime
            if (close[i] > donch_high_20_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and 
                vol_ratio_aligned[i] > 1.2):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian low with volume AND high volatility regime
            elif (close[i] < donch_low_20_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and 
                  vol_ratio_aligned[i] > 1.2):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price retracement to midpoint of Donchian channel
            donch_mid = (donch_high_20_aligned[i] + donch_low_20_aligned[i]) / 2
            if position == 1:
                if close[i] < donch_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian20_VolatilityBreakout_Volume"
timeframe = "12h"
leverage = 1.0