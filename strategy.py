#!/usr/bin/env python3
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
    
    # Get weekly data for regime detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly ADX(14) for trend strength
    # Calculate True Range
    tr1 = pd.Series(df_1w['high'] - df_1w['low'])
    tr2 = pd.Series(np.abs(df_1w['high'] - df_1w['close'].shift(1)))
    tr3 = pd.Series(np.abs(df_1w['low'] - df_1w['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = 0
    
    # Calculate Directional Movement
    up_move = df_1w['high'].diff()
    down_move = df_1w['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Daily Donchian Channel (20-period) for breakout signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Daily volume confirmation
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    volume_spike = df_1d['volume'].values > (vol_ma * 2.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        upper_band = donch_high_aligned[i]
        lower_band = donch_low_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        # Only trade when weekly ADX > 25 (trending market)
        if adx_val > 25:
            if position == 0:
                # Long: price breaks above upper Donchian band with volume spike
                if close[i] > upper_band and close[i-1] <= upper_band and vol_spike:
                    signals[i] = size
                    position = 1
                # Short: price breaks below lower Donchian band with volume spike
                elif close[i] < lower_band and close[i-1] >= lower_band and vol_spike:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long: price closes below lower Donchian band
                if close[i] < lower_band:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            elif position == -1:
                # Exit short: price closes above upper Donchian band
                if close[i] > upper_band:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
        else:
            # In ranging market (ADX <= 25), stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1d_DonchianBreakout_ADX_Volume_v1"
timeframe = "1d"
leverage = 1.0