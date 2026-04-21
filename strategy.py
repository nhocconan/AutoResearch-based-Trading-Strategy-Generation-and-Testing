#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeFilter
Hypothesis: 12h Donchian(20) breakout filtered by 1w EMA50 trend and volume confirmation.
In trending markets (price > EMA50_1w): breakout continuation (long above upper band, short below lower band).
In ranging markets: no entries to avoid false breakouts and reduce fee drag.
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to balance returns and fee drag.
Designed to work in both bull and bear markets by only trading with the 1w trend.
Timeframe: 12h, uses 1w HTF for trend filter.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 20-period Donchian channels on 12h timeframe ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 50-period EMA on 1w for trend filter ===
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume confirmation: current volume > 1.5x 20-period average ===
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
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_confirm = vol > 1.5 * vol_ma[i]
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Only trade in direction of 1w trend
            if price > upper_band[i] and price > ema_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif price < lower_band[i] and price < ema_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR) or trend reversal
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema_trend:  # Exit if price closes below 1w EMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR) or trend reversal
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema_trend:  # Exit if price closes above 1w EMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0