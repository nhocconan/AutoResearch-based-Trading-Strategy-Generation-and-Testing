#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: 12h Camarilla pivot (R1/S1) breakout filtered by 1d EMA50 trend and volume spike (1.8x average).
Enhanced with ATR-based dynamic position sizing (0.20-0.30) to reduce drawdown in volatile markets.
Long when price > R1 and above 1d EMA50; short when price < S1 and below 1d EMA50.
Volume confirmation reduces false breakouts. ATR(14) stoploss (2.5x) and discrete sizing (0.20, 0.25, 0.30).
Designed to work in both bull and bear markets via 1d trend alignment and strict entry filters.
Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot and EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    h3_1d = df_1d_close + 1.1 * range_1d
    l3_1d = df_1d_close - 1.1 * range_1d
    h4_1d = df_1d_close + 1.382 * range_1d
    l4_1d = df_1d_close - 1.382 * range_1d
    
    # Align 1d Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h ATR (14-period) for stoploss and dynamic sizing ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (50-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        h4 = h4_1d_aligned[i]
        l4 = l4_1d_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average (reduced from 2.0x to increase trades)
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Only enter in trending markets (price > 1d EMA50 for long, < for short)
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > r1) and (price > ema_trend) and volume_confirmed
            short_condition = (price < s1) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                # Dynamic sizing based on ATR volatility (lower ATR = larger position)
                atr_norm = atr[i] / price  # Normalized ATR
                if atr_norm < 0.015:  # Low volatility
                    size = 0.30
                elif atr_norm > 0.025:  # High volatility
                    size = 0.20
                else:  # Medium volatility
                    size = 0.25
                signals[i] = size
                position = 1
                entry_price = price
            elif short_condition:
                # Dynamic sizing based on ATR volatility (lower ATR = larger position)
                atr_norm = atr[i] / price  # Normalized ATR
                if atr_norm < 0.015:  # Low volatility
                    size = 0.30
                elif atr_norm > 0.025:  # High volatility
                    size = 0.20
                else:  # Medium volatility
                    size = 0.25
                signals[i] = -size
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr[i]:
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
                signals[i] = 0.25  # Maintain base size when holding
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
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
                signals[i] = -0.25  # Maintain base size when holding
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "12h"
leverage = 1.0