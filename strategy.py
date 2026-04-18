#!/usr/bin/env python3
"""
4h Donchian20 Breakout with Volume Spike and ADX Filter
Strategy: Enter long on Donchian20 upper break with volume spike and ADX>25,
          short on Donchian20 lower break with volume spike and ADX>25.
          Exit on opposite Donchian break or ADX<20.
          Uses 12h EMA34 for higher timeframe trend filter to avoid counter-trend trades.
          Designed for low trade frequency with clear breakout edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.absolute(high[1:] - close[:-1]), 
                    np.absolute(low[1:] - close[:-1]))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get 12h data for EMA34 trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        ema_34_12h = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Donchian upper break + volume spike + ADX>25 + above 12h EMA34
            if (price > donch_high[i] and volume_spike[i] and 
                adx_val > 25 and price > ema_34_12h):
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower break + volume spike + ADX>25 + below 12h EMA34
            elif (price < donch_low[i] and volume_spike[i] and 
                  adx_val > 25 and price < ema_34_12h):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: Donchian lower break or ADX<20
            if price < donch_low[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: Donchian upper break or ADX<20
            if price > donch_high[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADXFilter_12hEMA34"
timeframe = "4h"
leverage = 1.0