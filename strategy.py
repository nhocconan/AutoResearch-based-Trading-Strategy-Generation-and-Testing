#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla pivot (R1/S1) breakout filtered by 4h EMA50 trend and volume spike.
In trending markets (price > EMA50_4h): breakout continuation (long above R1, short below S1).
In ranging markets: no entries to avoid whipsaw. Uses volume confirmation (1.8x average) to filter false breakouts.
ATR(14) stoploss (1.5x) and discrete position sizing (0.20) to limit fee drag and drawdown.
Designed to work in both bull and bear markets by requiring strong trend alignment.
Timeframe: 1h, uses 4h HTF for trend filter.
Target: 60-150 total trades over 4 years = 15-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for EMA50 trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h OHLC for EMA50 trend ===
    df_4h_close = df_4h['close'].values
    ema_50_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 4h OHLC for Camarilla pivot calculation (based on previous 4h bar) ===
    df_4h_open = df_4h['open'].values
    df_4h_high = df_4h['high'].values
    df_4h_low = df_4h['low'].values
    df_4h_close = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    range_4h = df_4h_high - df_4h_low
    r1_4h = df_4h_close + 0.275 * range_4h
    s1_4h = df_4h_close - 0.275 * range_4h
    h3_4h = df_4h_close + 1.1 * range_4h
    l3_4h = df_4h_close - 1.1 * range_4h
    h4_4h = df_4h_close + 1.382 * range_4h
    l4_4h = df_4h_close - 1.382 * range_4h
    
    # Align 4h Camarilla levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    
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
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) 
            or np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_4h_aligned[i]
        s1 = s1_4h_aligned[i]
        h3 = h3_4h_aligned[i]
        l3 = l3_4h_aligned[i]
        h4 = h4_4h_aligned[i]
        l4 = l4_4h_aligned[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average (balanced filter)
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Only enter in trending markets (price > EMA50_4h for long, < for short)
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > r1) and (price > ema_trend) and volume_confirmed
            short_condition = (price < s1) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (1.5x ATR)
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at H3 (overbought)
            elif price > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss (1.5x ATR)
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at L3 (oversold)
            elif price < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0