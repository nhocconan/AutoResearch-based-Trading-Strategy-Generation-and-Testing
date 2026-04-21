#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeATRFilter_Tight_V1
Hypothesis: 4h Donchian(20) breakouts with volume confirmation (1.5x 20-bar avg volume) and ATR-based stoploss (2.0x ATR) will capture medium-term trends in BTC and ETH across bull and bear markets. Uses 1d EMA200 for regime filter (only long when price > EMA200, short when price < EMA200) to avoid counter-trend whipsaws. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for EMA200 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper (20-period high)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average volume
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Regime filter: 1d EMA200
        uptrend_regime = price > ema_200_1d_aligned[i]
        downtrend_regime = price < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend regime with volume
            if uptrend_regime and volume_ok:
                if price > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian low in downtrend regime with volume
            elif downtrend_regime and volume_ok:
                if price < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches Donchian low or stoploss
            if price <= donch_low[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches Donchian high or stoploss
            if price >= donch_high[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeATRFilter_Tight_V1"
timeframe = "4h"
leverage = 1.0