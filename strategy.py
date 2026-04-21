#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian20_Breakout_VolumeFilter
Hypothesis: On 6h timeframe, use 1d ADX to filter regimes (trending vs ranging).
In trending regime (ADX > 25): trade Donchian(20) breakouts with volume confirmation.
In ranging regime (ADX <= 25): fade Donchian(20) touches with volume confirmation.
This adaptive approach works in both bull/bear markets by adjusting to volatility regimes.
Uses weekly EMA34 as additional trend filter to avoid counter-trend trades in strong trends.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (daily for ADX, weekly for trend filter)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === Daily ADX (14-period) for regime detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Weekly EMA34 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 6h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume spike filter (20-period average) ===
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
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) 
            or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx = adx_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 0:
            # Regime-based logic
            if adx > 25:  # Trending regime
                # Breakout continuation with trend filter
                long_breakout = price > donchian_high[i]
                short_breakout = price < donchian_low[i]
                long_trend = price > ema_34_1w_aligned[i]
                short_trend = price < ema_34_1w_aligned[i]
                
                if long_breakout and long_trend and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_breakout and short_trend and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:  # Ranging regime (ADX <= 25)
                # Mean reversion at Donchian bands
                long_reversion = price <= donchian_low[i]  # Touch or break lower band
                short_reversion = price >= donchian_high[i]  # Touch or break upper band
                
                if long_reversion and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_reversion and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exits
            elif adx > 25:  # Trending: exit on opposite Donchian touch
                if price < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit at middle or opposite touch
                if price >= (donchian_high[i] + donchian_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exits
            elif adx > 25:  # Trending: exit on opposite Donchian touch
                if price > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit at middle or opposite touch
                if price <= (donchian_high[i] + donchian_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ADX_Regime_Donchian20_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0