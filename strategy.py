#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d ADX regime
# Long when Bull Power > 0 and Bear Power < 0 and 1d ADX > 25 (trending up)
# Short when Bull Power < 0 and Bear Power > 0 and 1d ADX > 25 (trending down)
# Exit when Elder Ray signals weaken or ADX < 20 (range regime)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_1dADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d indicators ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ADX for regime filter
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate +DM and -DM
    up_move = pd.Series(df_1d['high']).diff()
    down_move = pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    tr_smooth = pd.Series(atr_1d.values).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_13 = ema_13_aligned[i]
        curr_adx = adx_aligned[i]
        
        # Calculate Elder Ray components
        bull_power = curr_high - curr_ema_13
        bear_power = curr_ema_13 - curr_low
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending regime (ADX > 25)
            if curr_adx > 25:
                # Bullish: Bull Power > 0 and Bear Power < 0
                if bull_power > 0 and bear_power < 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Bull Power < 0 and Bear Power > 0
                elif bull_power < 0 and bear_power > 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Elder Ray weakens or regime changes to range
            if bull_power <= 0 or bear_power >= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Elder Ray weakens or regime changes to range
            if bull_power >= 0 or bear_power <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals