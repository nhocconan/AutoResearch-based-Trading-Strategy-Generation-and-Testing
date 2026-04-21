#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeATRFilter_Tight_V1
Hypothesis: Donchian(20) breakouts with volume confirmation and ATR-based stoploss on 4h timeframe work for BTC and ETH in both bull and bear markets. The strategy uses 1d ATR for volatility filter and 1w EMA200 for long-term trend. Target: 20-50 trades/year per symbol (80-200 over 4 years). Tight entry conditions to avoid overtrading and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data once for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data once for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter (avoid low volatility chop)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1w EMA200 for long-term trend filter (avoid counter-trend in strong trends)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 4h Donchian(20) channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 20-period average (approx 10 days on 4h)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr_1d_aligned[i] > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-100):i+1])
        
        # Long-term trend filter: only trade in direction of 1w EMA200
        uptrend_1w = close[i] > ema_200_1w_aligned[i]
        downtrend_1w = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume and vol filter
            if uptrend_1w and volume_ok and vol_filter:
                if price > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian low in downtrend with volume and vol filter
            elif downtrend_1w and volume_ok and vol_filter:
                if price < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches Donchian low or stoploss
            if price <= donch_low[i] or price < prices['close'].iloc[i-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches Donchian high or stoploss
            if price >= donch_high[i] or price > prices['close'].iloc[i-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeATRFilter_Tight_V1"
timeframe = "4h"
leverage = 1.0