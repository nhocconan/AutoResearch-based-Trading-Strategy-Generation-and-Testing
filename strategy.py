#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d ATR-scaled volume spike filter and 1w ADX trend filter.
    # Long when price breaks above Donchian(20) high + volume/ATR > 2.0x 20-period average + 1w ADX > 25 (trending).
    # Short when price breaks below Donchian(20) low + volume/ATR > 2.0x 20-period average + 1w ADX > 25 (trending).
    # Exit when price crosses Donchian(20) midpoint.
    # Uses volume spike to confirm breakout strength and 1w ADX to ensure we only trade in strong trends.
    # Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1d data for ATR-scaled volume filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate True Range (TR) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR(14) on 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume (volume / ATR) on 1d
    vol_atr_ratio_1d = volume_1d / np.maximum(atr_1d, 1e-10)
    
    # Align HTF ATR-scaled volume to 6h timeframe
    vol_atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_atr_ratio_1d)
    
    # Get 1w data for ADX trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range (TR) on 1w
    tr1_1w = np.abs(high_1w - low_1w)
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr1_1w[0]  # First period
    
    # Calculate ATR(14) on 1w
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI for ADX on 1w
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    tr_1w_smooth = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di_1w = 100 * plus_dm_smooth / np.maximum(tr_1w_smooth, 1e-10)
    minus_di_1w = 100 * minus_dm_smooth / np.maximum(tr_1w_smooth, 1e-10)
    
    # Calculate ADX
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / np.maximum(plus_di_1w + minus_di_1w, 1e-10)
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align HTF ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_atr_ratio_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume/ATR ratio > 2.0x 20-period average (using current value vs EMA)
        vol_atr_ma_1d = pd.Series(vol_atr_ratio_1d_aligned).ewm(span=20, adjust=False).mean().values
        volume_confirm = vol_atr_ratio_1d_aligned[i] > 2.0 * vol_atr_ma_1d[i]
        
        # Trend filter: 1w ADX > 25 indicates strong trend
        trend_filter = adx_1w_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry conditions: breakout + volume + trend
        long_signal = long_breakout and volume_confirm and trend_filter
        short_signal = short_breakout and volume_confirm and trend_filter
        
        # Exit conditions: price crosses Donchian midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
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

name = "6h_1d_1w_donchian_vol_atr_adx_v1"
timeframe = "6h"
leverage = 1.0