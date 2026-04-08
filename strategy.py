#!/usr/bin/env python3
# 4h_1d_adx_donchian_volume_v1
# Hypothesis: Trade Donchian channel breakouts on 4h with 1d ADX trend filter and volume confirmation.
# Enter long when price breaks above 20-period Donchian high with 1d ADX > 25 and volume surge.
# Enter short when price breaks below 20-period Donchian low with 1d ADX > 25 and volume surge.
# Exit when price crosses the Donchian midline (10-period EMA) or ADX weakens.
# Trend filter avoids whipsaws in sideways markets. Volume confirms breakout strength.
# Target: 20-40 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_adx_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    period = 14
    atr_1d = np.zeros_like(tr)
    atr_1d[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    plus_di_1d = 100 * np.where(atr_1d != 0, 
                                np.convolve(plus_dm, np.ones(period)/period, mode='full')[:len(atr_1d)] / atr_1d, 0)
    minus_di_1d = 100 * np.where(atr_1d != 0,
                                np.convolve(minus_dm, np.ones(period)/period, mode='full')[:len(atr_1d)] / atr_1d, 0)
    dx_1d = 100 * np.where((plus_di_1d + minus_di_1d) != 0,
                          np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    adx_1d = np.zeros_like(dx_1d)
    adx_1d[period-1] = np.mean(dx_1d[:period])
    for i in range(period, len(dx_1d)):
        adx_1d[i] = (adx_1d[i-1] * (period-1) + dx_1d[i]) / period
    
    # Pad arrays to match original length
    plus_di_1d_pad = np.zeros(len(high_1d))
    minus_di_1d_pad = np.zeros(len(high_1d))
    adx_1d_pad = np.zeros(len(high_1d))
    plus_di_1d_pad[1:] = plus_di_1d
    minus_di_1d_pad[1:] = minus_di_1d
    adx_1d_pad[1:] = adx_1d
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_pad)
    
    # 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: 4h volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below Donchian midline OR ADX weakens (< 20)
            if close[i] < donch_mid[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian midline OR ADX weakens (< 20)
            if close[i] > donch_mid[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above Donchian high with strong trend and volume surge
            if (high[i] > donch_high[i-1] and  
                adx_1d_aligned[i] > 25 and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Break below Donchian low with strong trend and volume surge
            elif (low[i] < donch_low[i-1] and 
                  adx_1d_aligned[i] > 25 and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals