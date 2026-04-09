#!/usr/bin/env python3
# 12h_donchian_1d_volatility_v1
# Hypothesis: 12h strategy using Donchian(20) breakout with volume confirmation and 1d ATR volatility filter.
# Enters long on upper band breakout, short on lower band breakout. Uses 1d ATR to normalize position size
# and avoid choppy markets. Designed for low trade frequency (target: 50-150 total trades over 4 years)
# to minimize fee drag. Works in bull/bear by using volatility filter to avoid false breakouts in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1d_volatility_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR (14-period) on 1d
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian channels (20-period) on 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_up = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_up + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(donchian_up[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period MA (avoid low volatility choppy periods)
        atr_ma = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
        volatility_filter = atr_1d_aligned[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        if not volatility_filter:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below mid-band OR volatility drops
            if close[i] < donchian_mid[i] or not volatility_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above mid-band OR volatility drops
            if close[i] > donchian_mid[i] or not volatility_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed[i] and volatility_filter:
                # Long conditions: price breaks above upper Donchian band
                if close[i] > donchian_up[i]:
                    position = 1
                    signals[i] = 0.25
                # Short conditions: price breaks below lower Donchian band
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals