#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX regime filter and volume confirmation
# Long when price breaks above 20-period 12h Donchian high AND 1w ADX > 20 (trending) AND 1w volume > 1.2x 20-period volume SMA
# Short when price breaks below 20-period 12h Donchian low AND 1w ADX > 20 AND 1w volume > 1.2x 20-period volume SMA
# Uses discrete position size 0.25 to limit drawdown. Target: 50-150 total trades over 4 years (12-37/year)

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
    
    # Get 12h data once before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1w data once before loop for ADX and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Donchian Channels (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian High = rolling max of high over 20 periods
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian Low = rolling min of low over 20 periods
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # === 1w Indicator: ADX (14-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === 1w Indicator: Volume SMA (20-period) for confirmation ===
    volume_1w = df_1w['volume'].values
    vol_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma_20_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (need 20 for Donchian, 14 for ADX, 20 for volume SMA)
    warmup = 40
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1w volume (aligned)
        vol_1w_series = df_1w['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_series)
        if np.isnan(vol_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1w volume > 1.2x 20-period 1w volume SMA
        vol_threshold = vol_sma_20_1w_aligned[i] * 1.2
        vol_confirm = vol_1w_aligned[i] > vol_threshold
        
        # ADX filter: trending market (ADX > 20)
        adx_trend = adx_1w_aligned[i] > 20
        
        # === LONG CONDITIONS ===
        # Price breaks above Donchian high AND trending market AND volume confirmation
        if (close[i] > donch_high_12h_aligned[i]) and adx_trend and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below Donchian low AND trending market AND volume confirmation
        elif (close[i] < donch_low_12h_aligned[i]) and adx_trend and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1wADX_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0