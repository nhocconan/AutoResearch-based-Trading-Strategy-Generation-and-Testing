#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX regime filter
# Long when price breaks above 20-period Donchian high + 1d ADX > 25 + volume > 1.3x 20-period avg
# Short when price breaks below 20-period Donchian low + 1d ADX > 25 + volume > 1.3x 20-period avg
# Uses 1d ADX for regime filtering (trending markets only) and Donchian channels for breakout signals
# Discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Works in both bull and bear markets by only trading strong trends with volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ADX (14-period) for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    atr_smooth = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di = np.where(atr_smooth != 0, 100 * (plus_dm_smooth / atr_smooth), 0)
    minus_di = np.where(atr_smooth != 0, 100 * (minus_dm_smooth / atr_smooth), 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    # Donchian high: highest high over last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 1d data for ADX (14+14+14 = ~42 periods) + Donchian(20) + volume(20) + buffer
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           trending and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             trending and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dADX_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0