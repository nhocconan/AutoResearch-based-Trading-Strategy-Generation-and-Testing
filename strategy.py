#!/usr/bin/env python3
"""
4h_ADX_EMA_Crossover_v1
4-hour strategy using EMA crossovers with ADX filter and volume confirmation.
Enters long when EMA21 crosses above EMA50 with ADX > 25 and volume above average.
Enters short when EMA21 crosses below EMA50 with ADX > 25 and volume above average.
Exits when opposite crossover occurs or ADX falls below 20.
Uses 1d ADX and volume for regime filter to avoid whipsaws in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === EMA21 and EMA50 on 4h ===
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Daily ADX for Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    plus_dm[1:] = np.maximum(high_1d[1:] - high_1d[:-1], 0)
    minus_dm[1:] = np.maximum(low_1d[:-1] - low_1d[1:], 0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1d[0] - low_1d[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Daily Volume for Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current day's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.3 * vol_ma_1d_aligned[i]
        
        # Regime filter: only trade when ADX > 25 (trending)
        trending = adx_1d_aligned[i] > 25
        # Exit regime filter: ADX < 20 indicates ranging/choppy
        ranging = adx_1d_aligned[i] < 20
        
        # EMA crossover signals
        ema_cross_up = ema21[i] > ema50[i] and ema21[i-1] <= ema50[i-1]
        ema_cross_down = ema21[i] < ema50[i] and ema21[i-1] >= ema50[i-1]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: EMA21 crosses above EMA50 with volume and trend
            if ema_cross_up and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: EMA21 crosses below EMA50 with volume and trend
            elif ema_cross_down and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: EMA21 crosses below EMA50 OR ADX drops to ranging
            if ema_cross_down or ranging:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA21 crosses above EMA50 OR ADX drops to ranging
            if ema_cross_up or ranging:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_EMA_Crossover_v1"
timeframe = "4h"
leverage = 1.0