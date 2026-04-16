#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h ADX regime filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + 12h ADX > 20
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + 12h ADX > 20
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Donchian channels provide clear breakout levels; volume confirmation reduces false signals
# 12h ADX ensures we only trade when higher timeframe is trending (works in both bull and bear)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: ADX (14-period) for trend strength ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.roll(low_12h, 1) - low_12h
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    atr_smooth = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI (avoid division by zero)
    plus_di = np.where(atr_smooth != 0, 100 * (plus_dm_smooth / atr_smooth), 0.0)
    minus_di = np.where(atr_smooth != 0, 100 * (minus_dm_smooth / atr_smooth), 0.0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 4h Indicators: Donchian Channel (20-period) and Volume SMA ===
    # Donchian Channel
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # 20 for Donchian + 14+14 for ADX + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # ADX filter: only trade when trending (ADX > 20)
        trending = adx_12h_aligned[i] > 20
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. 12h ADX > 20 (trending market on higher timeframe)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           trending and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. 12h ADX > 20 (trending market on higher timeframe)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             trending and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hADX_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0