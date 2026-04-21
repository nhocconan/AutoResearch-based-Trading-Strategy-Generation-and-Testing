#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla pivot (R1/S1) breakout filtered by 12h EMA50 trend and volume spike.
In trending markets (price > EMA50_12h for long, < for short): breakout continuation (long above R1, short below S1).
Volume confirmation (1.8x average) filters false breakouts. ATR(14) stoploss (2.0x) and discrete sizing (0.25).
Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year). Works in bull/bear via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA50 trend and Camarilla pivot)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h OHLC for EMA50 trend ===
    df_12h_close = df_12h['close'].values
    ema_50_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h OHLC for Camarilla pivot calculation (based on previous 12h bar) ===
    df_12h_open = df_12h['open'].values
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    range_12h = df_12h_high - df_12h_low
    r1_12h = df_12h_close + 0.275 * range_12h
    s1_12h = df_12h_close - 0.275 * range_12h
    h3_12h = df_12h_close + 1.1 * range_12h
    l3_12h = df_12h_close - 1.1 * range_12h
    h4_12h = df_12h_close + 1.382 * range_12h
    l4_12h = df_12h_close - 1.382 * range_12h
    
    # Align 12h Camarilla levels to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
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
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_12h_aligned[i]
        s1 = s1_12h_aligned[i]
        h3 = h3_12h_aligned[i]
        l3 = l3_12h_aligned[i]
        h4 = h4_12h_aligned[i]
        l4 = l4_12h_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average (tight filter)
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Only enter in trending markets (price > EMA50_12h for long, < for short)
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > r1) and (price > ema_trend) and volume_confirmed
            short_condition = (price < s1) and (price < ema_trend) and volume_confirmed
            
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
            # Mean reversion exit at H4 (extreme overbought)
            elif price > h4:
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
            # Mean reversion exit at L4 (extreme oversold)
            elif price < l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0