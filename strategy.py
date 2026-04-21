#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend_A
Hypothesis: 4h Donchian(20) breakout with volume confirmation (1.8x average) and 1d EMA50 trend filter.
Long when price breaks above upper channel AND price > EMA50_1d AND volume spike.
Short when price breaks below lower channel AND price < EMA50_1d AND volume spike.
ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to limit fee drag.
Designed to capture strong trends in both bull and bear markets by requiring volume and trend alignment.
Timeframe: 4h, uses 1d HTF for trend filter.
Target: 75-200 total trades over 4 years = 19-50/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    df_1d_close = df_1d['close'].values
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper channel: highest high of last 20 bars
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 bars
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_1d_aligned[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average (balanced filter)
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Breakout entry with volume and trend confirmation
            long_condition = (price > upper[i]) and (price > ema_trend) and volume_confirmed
            short_condition = (price < lower[i]) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at upper channel (overbought)
            elif price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at lower channel (oversold)
            elif price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend_A"
timeframe = "4h"
leverage = 1.0