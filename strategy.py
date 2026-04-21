#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Volume_ATR_v1
Hypothesis: Breakout of Camarilla R4/S4 levels on 1d with 1w trend filter and volume confirmation.
Works in bull/bear: In uptrend (1w EMA rising), buy R4 breakout; in downtrend (1w EMA falling), sell S4 breakout.
Uses 1w EMA for trend, volume spike for confirmation, ATR for dynamic position sizing and stoploss.
Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R4, S4 (extreme breakout levels)
    rang = prev_high - prev_low
    r4 = prev_close + rang * 6.0 / 12
    s4 = prev_close - rang * 6.0 / 12
    
    # Align to 1d timeframe (no shift needed as we use previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d ATR(14) for stoploss and position sizing
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Load 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Determine 1w trend: EMA20 rising/falling
        if i > 0:
            ema_rising = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
            ema_falling = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        else:
            ema_rising = True
            ema_falling = False
        
        if position == 0:
            # Long entry: price breaks above R4 AND 1w uptrend AND volume spike
            if (price > r4_aligned[i] and 
                ema_rising and 
                volume_ok):
                signals[i] = 0.30
                position = 1
                entry_price = price
                atr_at_entry = atr_1d_aligned[i]
            # Short entry: price breaks below S4 AND 1w downtrend AND volume spike
            elif (price < s4_aligned[i] and 
                  ema_falling and 
                  volume_ok):
                signals[i] = -0.30
                position = -1
                entry_price = price
                atr_at_entry = atr_1d_aligned[i]
        
        elif position == 1:
            # Long exit: stoploss hit or trend reversal
            stoploss = entry_price - 2.5 * atr_at_entry
            if price < stoploss or not ema_rising:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: stoploss hit or trend reversal
            stoploss = entry_price + 2.5 * atr_at_entry
            if price > stoploss or not ema_falling:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout_Volume_ATR_v1"
timeframe = "1d"
leverage = 1.0