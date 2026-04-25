#!/usr/bin/env python3
"""
6h Elder Ray Index + 1d ADX Regime + Volume Confirmation
Hypothesis: Elder Ray (Bull/Bear Power) identifies momentum strength while 1d ADX filters regime (trending vs ranging). In trending markets (ADX>25), we take Elder Ray signals in trend direction. In ranging markets (ADX<20), we fade extreme Elder Ray readings. Volume confirmation reduces false signals. Designed for 6h to target 12-37 trades/year, works in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray (standard setting)
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray to 6h timeframe (with 1-bar delay for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d ADX for regime filter
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_smooth = pd.Series(atr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 14, 13)  # volume MA, ADX, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        adx = adx_aligned[i]
        
        # Regime classification
        trending = adx > 25
        ranging = adx < 20
        
        if position == 0:
            # Look for entry signals
            long_entry = False
            short_entry = False
            
            if trending:
                # In trending markets: follow Elder Ray momentum
                long_entry = (bull_power > 0) and vol_spike
                short_entry = (bear_power < 0) and vol_spike
            elif ranging:
                # In ranging markets: fade extreme Elder Ray readings
                # Long when bear power is extremely negative (oversold)
                # Short when bull power is extremely positive (overbought)
                long_entry = (bear_power < -np.std(bear_power_aligned[max(0, i-50):i])) and vol_spike
                short_entry = (bull_power > np.std(bull_power_aligned[max(0, i-50):i])) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Elder Ray turns negative OR loss of volume momentum
            if (bull_power <= 0) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Elder Ray turns positive OR loss of volume momentum
            if (bear_power >= 0) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0