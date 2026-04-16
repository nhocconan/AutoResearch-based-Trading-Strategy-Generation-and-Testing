#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX regime filter
# Long when price breaks above 20-period Donchian high + 1d ADX > 25 + volume > 1.5x 20-period avg
# Short when price breaks below 20-period Donchian low + 1d ADX > 25 + volume > 1.5x 20-period avg
# Uses 1d ADX calculated from prior 1d OHLC, aligned to 12h bars
# Discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Donchian breakouts capture strong trends; ADX filter ensures we only trade in trending conditions
# Volume confirmation reduces false breakouts
# Works in both bull and bear markets by capturing directional moves with trend filter

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
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 13:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[i-13:i+1])
    
    # +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR (using Wilder's smoothing)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    atr_smooth = np.zeros_like(atr)
    
    # Initial values (simple average)
    if len(plus_dm) >= 14:
        plus_dm_smooth[13] = np.mean(plus_dm[:14])
        minus_dm_smooth[13] = np.mean(minus_dm[:14])
        atr_smooth[13] = np.mean(atr[13:27]) if len(atr) >= 27 else np.nan
    
    # Wilder's smoothing
    for i in range(14, len(plus_dm)):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
        if not np.isnan(atr_smooth[i-1]):
            atr_smooth[i] = atr_smooth[i-1] - (atr_smooth[i-1] / 14) + atr[i]
        else:
            atr_smooth[i] = np.nan
    
    # +DI and -DI
    plus_di = np.where(atr_smooth != 0, 100 * (plus_dm_smooth / atr_smooth), 0)
    minus_di = np.where(atr_smooth != 0, 100 * (minus_dm_smooth / atr_smooth), 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    
    # ADX smoothing (14-period)
    for i in range(14, len(dx)):
        if i == 14:
            adx[i] = np.mean(dx[1:15]) if not np.any(np.isnan(dx[1:15])) else np.nan
        elif not np.isnan(adx[i-1]):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
        else:
            adx[i] = np.nan
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === LT Indicators: Donchian Channel (20-period) ===
    # Donchian high: highest high over past 20 periods
    # Donchian low: lowest low over past 20 periods
    donchian_high = np.full_like(close, np.nan)
    donchian_low = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_sma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 1d ADX (14+14+14 = ~42 periods) + Donchian(20) + volume(20) + buffer
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
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

name = "12h_Donchian20_1dADX_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0