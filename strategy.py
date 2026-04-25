#!/usr/bin/env python3
"""
6h_ADX_DMI_Trend_With_Volume_Confirmation
Hypothesis: On 6h timeframe, enter long when ADX>25 (trending) + +DI>-DI (bullish momentum) + volume>1.5x 20-period average. Enter short when ADX>25 + -DI>+DI (bearish momentum) + volume confirmation. Exit when ADX<20 (trend weakening) or opposite DI crossover. Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades. Designed for ~60-120 trades over 4 years (15-30/year) via strict ADX/DMI trend conditions with volume confirmation and HTF alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need 50 for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX and DMI on 6h data (primary timeframe)
    adx_period = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Plus Directional Movement
    plus_dm = high - np.roll(high, 1)
    minus_dm = np.roll(low, 1) - low
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero when both DI are zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, adx_period*2, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (ADX>25) and aligned with 1d trend
            if adx_val > 25 and close[i] > ema_trend:  # 1d uptrend filter
                # Long: bullish momentum (+DI > -DI) with volume confirmation
                long_signal = (plus_di_val > minus_di_val) and vol_regime[i]
            elif adx_val > 25 and close[i] < ema_trend:  # 1d downtrend filter
                # Short: bearish momentum (-DI > +DI) with volume confirmation
                short_signal = (minus_di_val > plus_di_val) and vol_regime[i]
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: 
            # 1. ADX < 20 (trend weakening)
            # 2. Bearish crossover (-DI > +DI)
            if adx_val < 20 or minus_di_val > plus_di_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. ADX < 20 (trend weakening)
            # 2. Bullish crossover (+DI > -DI)
            if adx_val < 20 or plus_di_val > minus_di_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_Trend_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0