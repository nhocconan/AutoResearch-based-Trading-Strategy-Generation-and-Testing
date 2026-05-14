#!/usr/bin/env python3
"""
4h_Vortex_Regime_Breakout_v1
Hypothesis: On 4h timeframe, Vortex indicator detects trend initiation, combined with 12h EMA trend filter and volume spike confirmation. Designed for low trade frequency (<50/year) to minimize fee drag and work in both bull (trend follow) and bear (mean revert in chop) regimes via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12-hour EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === ADX for regime filter (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Vortex Indicator (14-period) ===
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    sum_vm_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # === Volume spike filter ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_12h_aligned[i]
        adx_val = adx[i]
        vi_p = vi_plus[i]
        vi_m = vi_minus[i]
        vol_spike = vol_ratio[i] > 1.5  # 50% above average volume
        
        if position == 0:
            # Long: VI+ > VI- (bullish vortex) + above 12h EMA34 + ADX > 20 (trending) + volume spike
            if vi_p > vi_m and price_close > ema_34 and adx_val > 20 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: VI- > VI+ (bearish vortex) + below 12h EMA34 + ADX > 20 (trending) + volume spike
            elif vi_m > vi_p and price_close < ema_34 and adx_val > 20 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit conditions: trend weakening (ADX < 20) or vortex reversal
            if position == 1:
                if adx_val < 20 or vi_m > vi_p:  # trend weakening or bearish vortex
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if adx_val < 20 or vi_p > vi_m:  # trend weakening or bullish vortex
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Vortex_Regime_Breakout_v1"
timeframe = "4h"
leverage = 1.0