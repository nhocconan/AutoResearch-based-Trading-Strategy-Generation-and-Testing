#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_VolumeFilter
Hypothesis: 6h Donchian(20) breakout filtered by weekly Camarilla pivot direction and volume spike.
In bullish weekly context (close > weekly R3): long breakout above Donchian upper band.
In bearish weekly context (close < weekly S3): short breakout below Donchian lower band.
Volume confirmation (>1.8x average) filters false breakouts. Designed to work in both bull and bear markets
by requiring alignment with weekly pivot structure. Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for Camarilla pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly OHLC for Camarilla pivot calculation (based on previous weekly bar) ===
    df_1w_open = df_1w['open'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    range_1w = df_1w_high - df_1w_low
    r3_1w = df_1w_close + 1.1 * range_1w
    s3_1w = df_1w_close - 1.1 * range_1w
    r4_1w = df_1w_close + 1.382 * range_1w
    s4_1w = df_1w_close - 1.382 * range_1w
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 6h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower bands (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) 
            or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        donch_upper = donch_high[i]
        donch_lower = donch_low[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        r4 = r4_1w_aligned[i]
        s4 = s4_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Bullish weekly context: close > weekly R3
            # Bearish weekly context: close < weekly S3
            weekly_bullish = price > r3
            weekly_bearish = price < s3
            
            # Long breakout: price > Donchian upper band in bullish weekly context
            long_condition = (price > donch_upper) and weekly_bullish and volume_confirmed
            # Short breakout: price < Donchian lower band in bearish weekly context
            short_condition = (price < donch_lower) and weekly_bearish and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (1.5x ATR)
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Weekly context reversal exit
            elif price < s3:  # Weekly turned bearish
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at weekly R4 (overbought extreme)
            elif price > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (1.5x ATR)
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Weekly context reversal exit
            elif price > r3:  # Weekly turned bullish
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at weekly S4 (oversold extreme)
            elif price < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0