#!/usr/bin/env python3
"""
6h_ElderRay_Regime_ADX_v1
Hypothesis: On 6h timeframe, Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) combined with ADX(14) regime filter and 1d trend direction captures strong trending moves while avoiding whipsaw in ranging markets. 
In trending regime (ADX > 25), trade in direction of 1d EMA50 trend: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling. 
In ranging regime (ADX < 20), fade extremes: long when Bull Power crosses above -ATR(10) and price > 6h VWAP, short when Bear Power crosses below ATR(10) and price < 6h VWAP.
Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Elder Ray Index ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h ADX(14) for regime filter ===
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr_ma = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / atr_ma
    minus_di = 100 * minus_dm_ma / atr_ma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 6h VWAP for mean reversion entries ===
    volume = prices['volume'].values
    vwap_num = pd.Series(close * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_num / vwap_den
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 2 days (8 * 6h = 48h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(atr[i]) or np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        adx_val = adx[i]
        atr_val = atr[i]
        vwap_val = vwap[i]
        
        # Regime classification
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            if is_trending:
                # Trending regime: follow 1d EMA50 trend
                is_bull_trend = price > ema_50
                is_bear_trend = price < ema_50
                
                # Long conditions: Bull Power > 0 and rising (vs previous bar)
                long_condition = (bull > 0) and (i > 0 and bull > bull_power[i-1])
                # Short conditions: Bear Power < 0 and falling (vs previous bar)
                short_condition = (bear < 0) and (i > 0 and bear < bear_power[i-1])
                
                if is_bull_trend and long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif is_bear_trend and short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
            
            elif is_ranging:
                # Ranging regime: fade extremes at power/ATR thresholds
                # Long when Bull Power crosses above -ATR(10) and price > VWAP
                long_condition = (bull > -atr_val) and (i > 0 and bull_power[i-1] <= -atr_val) and (price > vwap_val)
                # Short when Bear Power crosses below ATR(10) and price < VWAP
                short_condition = (bear < atr_val) and (i > 0 and bear_power[i-1] >= atr_val) and (price < vwap_val)
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_ADX_v1"
timeframe = "6h"
leverage = 1.0