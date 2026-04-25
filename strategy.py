#!/usr/bin/env python3
"""
6h ADX + Donchian(20) Breakout + Volume Confirmation
Hypothesis: In strong trends (ADX>25), Donchian breakouts capture momentum with volume confirmation.
ADX filter avoids whipsaws in ranging markets. Works in bull/bear via trend direction from Donchian midpoint.
Target: 12-37 trades/year on 6h timeframe.
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
    
    # Get 12h data for ADX trend filter (more stable than 6h ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).diff().abs()
    tr2 = (pd.Series(high_12h) - pd.Series(close_12h).shift()).abs()
    tr3 = (pd.Series(low_12h) - pd.Series(close_12h).shift()).abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = pd.Series(high_12h).diff()
    down_move = pd.Series(low_12h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h.values).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Donchian channels on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate ATR(14) for stoploss and volume average
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_aligned[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        mid_donchian = donchian_mid[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirmed = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian AND strong trend AND volume confirmed
            long_condition = (curr_high > upper_donchian) and strong_trend and volume_confirmed
            # Short: price breaks below lower Donchian AND strong trend AND volume confirmed
            short_condition = (curr_low < lower_donchian) and strong_trend and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or trend weakness (ADX < 20) or price below midpoint
            if (curr_close <= entry_price - 2.0 * atr_val) or (adx_val < 20) or (curr_close < mid_donchian):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or trend weakness (ADX < 20) or price above midpoint
            if (curr_close >= entry_price + 2.0 * atr_val) or (adx_val < 20) or (curr_close > mid_donchian):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Donchian_Breakout_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0