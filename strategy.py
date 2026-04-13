#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter (ADX>25).
    # Long when price breaks above Camarilla H3 + 4h volume > 1.2x 20-period MA + 1d ADX>25.
    # Short when price breaks below Camarilla L3 + 4h volume > 1.2x 20-period MA + 1d ADX>25.
    # Exit when price crosses back to Camarilla Pivot point.
    # Uses Camarilla for intraday structure, volume for confirmation, ADX for trend regime.
    # Session filter: 08-20 UTC to avoid low-liquidity hours.
    # Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    camarilla_h3 = camarilla_pivot + camarilla_range * 1.1 / 4.0
    camarilla_l3 = camarilla_pivot - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Get 1d data for ADX trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate ATR (14) on 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM on 1d
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate +DI and -DI (14) on 1d
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / np.maximum(atr_1d, 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / np.maximum(atr_1d, 1e-10)
    
    # Calculate DX and ADX (14) on 1d
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / np.maximum(plus_di_1d + minus_di_1d, 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.2x 20-period average
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_spike = volume_4h_aligned[i] > 1.2 * vol_ma_4h_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1d_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Exit conditions: price crosses back to Camarilla Pivot
        long_exit = close[i] < camarilla_pivot_aligned[i]
        short_exit = close[i] > camarilla_pivot_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.20
        
        # Entry conditions
        if long_breakout and volume_spike and trend_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and volume_spike and trend_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_volume_adx_v1"
timeframe = "1h"
leverage = 1.0