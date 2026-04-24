#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w ADX regime filter.
- Primary timeframe: 1d for execution and signal generation.
- HTF: 1w for Donchian channels (structure), volume MA (confirmation), and ADX (trend strength).
- ADX > 25 indicates trending market (favor breakouts), ADX < 20 indicates ranging (avoid breakouts).
- Entry: Long when price breaks above 20-period 1w Donchian HIGH AND ADX > 25 AND volume > 1.5 * 20w volume MA.
         Short when price breaks below 20-period 1w Donchian LOW AND ADX > 25 AND volume > 1.5 * 20w volume MA.
- Exit: Opposite Donchian breakout or ADX drops below 20 (regime shift to ranging).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 20-60 total trades over 4 years (5-15/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels, volume MA, and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1w
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA on 1w
    volume_ma = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX (14-period) on 1w
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1d
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need enough 1w bars for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_ma = volume_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals: only in trending regime (ADX > 25)
            if adx_val > 25 and volume[i] > 1.5 * vol_ma:
                # Bullish breakout: price breaks above upper Donchian band
                if curr_high > upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian band
                elif curr_low < lower:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian band OR ADX drops to ranging
            if curr_low < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian band OR ADX drops to ranging
            if curr_high > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wVolumeSpike_1wADXRegime_v1"
timeframe = "1d"
leverage = 1.0