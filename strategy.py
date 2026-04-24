#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
- Uses Donchian(20) from prior completed 1d candles for breakout levels.
- Breakout above upper band or below lower band with volume > 2.0x 20-bar average.
- Trend filter: 1d ADX > 25 to ensure we only trade in trending markets (works in both bull/bear).
- Designed for 12h timeframe to capture medium-term breakouts with lower trade frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-30 trades/year (50-120 total over 4 years) to stay fee-efficient.
- Based on proven pattern: Donchian breakout + volume + trend filter showed strong performance in DB.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from prior completed 1d candles
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-period high, Lower band: 20-period low
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe (wait for 1d bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 1d ADX trend filter (14-period)
    # Calculate True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)
    
    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    
    # Calculate ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # ADX trend filter (> 25 indicates trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: breakout above upper band AND volume confirmation AND trending market
            if close[i] > donchian_upper_aligned[i] and volume_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND volume confirmation AND trending market
            elif close[i] < donchian_lower_aligned[i] and volume_confirm and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below lower band OR loss of trend
            if close[i] < donchian_lower_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above upper band OR loss of trend
            if close[i] > donchian_upper_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dADX_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0