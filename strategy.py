#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend filter (strong trend identification for BTC/ETH).
- Entry: Long when Williams %R < -80 (oversold) AND 1d ADX > 25 AND volume > 1.5 * 6h volume MA(20);
         Short when Williams %R > -20 (overbought) AND 1d ADX > 25 AND volume > 1.5 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via Williams %R midpoint (signal=0 when %R crosses -50).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies exhaustion points; 1d ADX ensures we only trade in strong trends to avoid chop.
- Volume confirmation adds conviction to reversal signals.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff(1).values
    down_move = pd.Series(low_1d).diff(1).values * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 6h data for volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # Calculate volume MA(20) on 6h
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R needs no extra delay
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 14+13, 20)  # Williams %R, ADX smoothing, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if Williams %R crosses -50 (midpoint)
        if position == 1:
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and ADX trend filter
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        strong_trend = adx_aligned[i] > 25
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and strong_trend:
                # Long: Oversold AND strong trend
                if oversold:
                    signals[i] = 0.25
                    position = 1
                # Short: Overbought AND strong trend
                elif overbought:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADXTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0