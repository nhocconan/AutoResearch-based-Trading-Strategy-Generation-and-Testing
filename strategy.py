#!/usr/bin/env python3
"""
1d_KAMA_Regime_ADX_v2
Hypothesis: On 1d timeframe, KAMA (ER=10) identifies adaptive trend direction while weekly ADX(14) filters for trending regimes (ADX > 25). 
Entry occurs when KAMA turns bullish/bearish AND weekly ADX confirms trending market. Exit when KAMA reverses or ADX drops below 20 (range regime).
Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years.
Works in both bull (trend following) and bear (trend following with shorts) markets via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend, 1d for ADX)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime (optional filter) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d ADX(14) for trend strength regime ===
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
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(atr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1d KAMA (ER=10) for adaptive trend ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.concatenate([np.full(10, np.nan), er])  # align with close
    
    # Smoothing Constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    
    for i in range(10, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction (1 = up, -1 = down)
    kama_dir = np.diff(kama, prepend=kama[0])
    kama_dir = np.sign(kama_dir)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(kama_dir[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1w_aligned[i]
        adx_val = adx_aligned[i]
        kama_val = kama[i]
        kama_direction = kama_dir[i]
        
        # Weekly trend filter (optional: only trade in direction of weekly trend)
        weekly_bull = price > weekly_ema
        weekly_bear = price < weekly_ema
        
        # Regime filters
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long conditions: KAMA turning up AND trending regime
            long_condition = (kama_direction == 1) and is_trending and weekly_bull
            # Short conditions: KAMA turning down AND trending regime
            short_condition = (kama_direction == -1) and is_trending and weekly_bear
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions: KAMA reversal OR ADX drops into ranging
            if position == 1:
                if (kama_direction == -1) or is_ranging:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (kama_direction == 1) or is_ranging:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_ADX_v2"
timeframe = "1d"
leverage = 1.0