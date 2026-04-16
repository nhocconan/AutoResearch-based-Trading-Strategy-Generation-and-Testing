#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX regime filter
# Long when price breaks above Donchian upper band (20-period high) + ADX > 25 + volume > 1.5x 20-period avg
# Short when price breaks below Donchian lower band (20-period low) + ADX > 25 + volume > 1.5x 20-period avg
# Uses 1d ADX for regime filter (trending market confirmation)
# Discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Donchian channels work in both bull and bear markets; ADX filter ensures we only trade in trending conditions

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
    if len(df_1d) < 50:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # === 1d Indicator: ADX (14-period) for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Convert to pandas Series for rolling operations
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range calculation
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # +DM and -DM calculation
    up_move = high_1d_series.diff()
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    atr_smooth = atr.rolling(window=14, min_periods=14).mean()
    
    # +DI and -DI calculation
    plus_di = 100 * (plus_dm_smooth / atr_smooth)
    minus_di = 100 * (minus_dm_smooth / atr_smooth)
    
    # DX and ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 4h timeframe (wait for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # === 4h Indicators: Donchian Channels (20-period) ===
    # Donchian upper band = 20-period high
    # Donchian lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 20 periods for Donchian + 14+14+14 for ADX + 20 for volume + buffer
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper band (20-period high)
        # 2. ADX > 25 (trending market)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and \
           trending and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower band (20-period low)
        # 2. ADX > 25 (trending market)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and \
             trending and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dADX_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0