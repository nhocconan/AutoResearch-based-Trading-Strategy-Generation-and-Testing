#!/usr/bin/env python3
# 4h_volatility_breakout_v1
# Hypothesis: Volatility breakouts with volume confirmation and multi-timeframe trend alignment.
# Long when: Price breaks above Donchian(20) high, volume > 2x 20-period average, and 1d EMA50 rising.
# Short when: Price breaks below Donchian(20) low, volume > 2x 20-period average, and 1d EMA50 falling.
# Exit: Opposite Donchian break or volatility contraction (ATR < 0.5 * 20-period ATR mean).
# Target: 20-40 trades/year with strict volatility/volume conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR for volatility measurement and exit
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, lookback)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_1d_50_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Volatility expansion condition (for exit)
        vol_expansion = atr[i] > 0.5 * np.nanmean(atr[max(0, i-20):i+1]) if not np.isnan(np.nanmean(atr[max(0, i-20):i+1])) else False
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR volatility contraction
            if close[i] < donchian_low[i] or not vol_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR volatility contraction
            if close[i] > donchian_high[i] or not vol_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high, volume surge, 1d EMA50 rising
            if (close[i] > donchian_high[i] and 
                vol_surge and 
                i > 0 and 
                ema_1d_50_aligned[i] > ema_1d_50_aligned[i-1]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, volume surge, 1d EMA50 falling
            elif (close[i] < donchian_low[i] and 
                  vol_surge and 
                  i > 0 and 
                  ema_1d_50_aligned[i] < ema_1d_50_aligned[i-1]):
                position = -1
                signals[i] = -0.25
    
    return signals