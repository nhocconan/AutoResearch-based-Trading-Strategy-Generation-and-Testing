#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + volume confirmation + ADX regime filter for trend strength
# Long when price breaks above 4h Donchian(20) high AND volume > 1.5x 20-period average AND ADX > 25 (trending market)
# Short when price breaks below 4h Donchian(20) low AND volume > 1.5x 20-period average AND ADX > 25
# Uses discrete position sizing (0.25) to control drawdown. Target: 50-150 total trades over 4 years

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
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    highest_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # === 4h Indicator: ADX (14-period) for trend strength ===
    # True Range calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_4h
    di_minus = 100 * dm_minus_smooth / atr_4h
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_4h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # === 4h Indicator: Volume SMA (20-period) for confirmation ===
    vol_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period 4h volume SMA
        vol_confirm = False
        if not np.isnan(volume[i]) and not np.isnan(vol_sma_20_aligned[i]):
            vol_threshold = vol_sma_20_aligned[i] * 1.5
            vol_confirm = volume[i] > vol_threshold
        
        # === LONG CONDITIONS ===
        # Price breaks above Donchian high AND volume confirmation AND ADX > 25 (trending)
        if (close[i] > highest_high_aligned[i]) and vol_confirm and (adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below Donchian low AND volume confirmation AND ADX > 25 (trending)
        elif (close[i] < lowest_low_aligned[i]) and vol_confirm and (adx_aligned[i] > 25):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0