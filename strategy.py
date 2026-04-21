#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop
Hypothesis: 1h Camarilla pivot R1/S1 breakout with volume confirmation (>1.3x 20-period volume MA) and 4h HTF trend filter (price > EMA34 for longs, < EMA34 for shorts). Uses 1d choppiness regime filter (CHOP > 61.8 = mean reversion, CHOP < 38.2 = trend) to adapt strategy. In choppy regimes: mean reversion at S1/R1 with tighter stops. In trending regimes: breakout continuation with momentum filter. Designed for 1h timeframe targeting 15-35 trades/year to minimize fee drag while capturing both bull and bear market moves via regime adaptation and HTF trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, 1d for regime)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 34 or len(df_1d) < 14:
        return np.zeros(n)
    
    # === 4h EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d Choppiness Index for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for choppy calculation
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum_1d = tr_1d.rolling(window=14, min_periods=14).sum().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_sum_1d / (highest_high_1d - lowest_low_1d)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 1h Indicators (primary timeframe) ===
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Calculate Camarilla pivot levels from previous day using 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_1h - low_1h)
    tr2 = pd.Series(np.abs(high_1h - np.roll(close_1h, 1)))
    tr3 = pd.Series(np.abs(low_1h - np.roll(close_1h, 1)))
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr_1h.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i])
            or np.isnan(ema_34_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1h[i]
        vol = volume_1h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        
        # Regime detection from 1d
        is_choppy = chop_1d_aligned[i] > 61.8  # mean reversion regime
        is_trending = chop_1d_aligned[i] < 38.2  # trend following regime
        
        if position == 0:
            # Long conditions
            long_breakout = price > s1_aligned[i]
            long_volume = vol_ok
            long_trend = price > ema_34_4h_aligned[i]  # 4h uptrend
            
            # Short conditions
            short_breakout = price < r1_aligned[i]
            short_volume = vol_ok
            short_trend = price < ema_34_4h_aligned[i]  # 4h downtrend
            
            if is_choppy:
                # In choppy market: mean reversion at S1/R1 with volume
                if long_breakout and long_volume:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                elif short_breakout and short_volume:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
            elif is_trending:
                # In trending market: breakout continuation with trend filter
                if long_breakout and long_volume and long_trend:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                elif short_breakout and short_volume and short_trend:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions
            elif (is_choppy and price < s1_aligned[i]) or \
                 (is_trending and price < ema_34_4h_aligned[i]) or \
                 not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions
            elif (is_choppy and price > r1_aligned[i]) or \
                 (is_trending and price > ema_34_4h_aligned[i]) or \
                 not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop"
timeframe = "1h"
leverage = 1.0