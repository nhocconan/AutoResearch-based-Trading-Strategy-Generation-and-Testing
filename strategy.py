#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume confirmation + ADX trend filter
# Long when price breaks above 4h Donchian high + 12h volume > 1.5x average + ADX > 25
# Short when price breaks below 4h Donchian low + 12h volume > 1.5x average + ADX > 25
# Exit when price crosses Donchian midpoint or volume drops below average
# Uses 4h timeframe to limit trades, targets 75-200 total over 4 years
# Works in both bull/bear markets by capturing breakouts with volume confirmation

name = "4h_donchian20_12h_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    donch_high = donch_high.values
    donch_low = donch_low.values
    donch_mid = donch_mid.values
    
    # 12h volume confirmation (get once before loop)
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean()
    vol_ma_12h = vol_ma_12h.values
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # ADX (14-period) from 4h for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_sum = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_sum = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Calculate DI and DX
    di_plus = 100 * dm_plus_sum / (tr_sum + 1e-10)
    di_minus = 100 * dm_minus_sum / (tr_sum + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx = adx.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_12h_aligned[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses midpoint or volume drops
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or volume[i] <= vol_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or volume[i] <= vol_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + trend
            # Long: price breaks above Donchian high + volume confirmation + ADX > 25
            if (close[i] > donch_high[i] and 
                volume[i] > 1.5 * vol_12h_aligned[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + ADX > 25
            elif (close[i] < donch_low[i] and 
                  volume[i] > 1.5 * vol_12h_aligned[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
    
    return signals