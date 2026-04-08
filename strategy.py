#!/usr/bin/env python3
# 4h_1d_1w_donchian_breakout_volume_trend_v1
# Hypothesis: Trade Donchian channel breakouts on 4h with 1d volume confirmation and 1w trend filter.
# In 1w uptrend: long on breakout above 20-period high, short on breakdown below 20-period low.
# In 1w downtrend: short on breakdown below 20-period low, long on breakout above 20-period high.
# Uses volume > 1.5x 20-period average to confirm breakouts and avoid false signals.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_donchian_breakout_volume_trend_v1"
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
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition (using daily volume MA aligned to 4h)
        vol_surge = volume[i] > 1.5 * vol_ma_20_1d_aligned[i] if vol_ma_20_1d_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price < Donchian low or weekly trend breaks (price < weekly EMA21)
            if close[i] < donchian_low[i] or close[i] < ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high or weekly trend breaks (price > weekly EMA21)
            if close[i] > donchian_high[i] or close[i] > ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > Donchian high with volume surge and weekly uptrend
            if (close[i] > donchian_high[i] and vol_surge and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < Donchian low with volume surge and weekly downtrend
            elif (close[i] < donchian_low[i] and vol_surge and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals