#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d ADX trend filter and 1d volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND ADX > 25 (trending) AND 1d volume > 1.5x 20-period average
# Short when Bear Power < 0 (close < EMA13) AND ADX > 25 (trending) AND 1d volume > 1.5x 20-period average
# Elder Ray measures bull/bear power relative to EMA13, providing clear trend strength signals
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranges
# Volume confirmation adds conviction to breakouts
# Discrete position sizing (0.25) to control drawdown. Target: 50-150 total trades over 4 years

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
    
    # Get 6h data once before loop for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data once before loop for ADX and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 6h Indicator: Elder Ray (Bull Power / Bear Power) ===
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA13 for 6h data
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_6h = high_6h - ema13_6h
    bear_power_6h = low_6h - ema13_6h
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # === 1d Indicator: ADX (14-period) for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        vol_confirm = False
        if not np.isnan(vol_1d_aligned[i]):
            vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
            vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # === LONG CONDITIONS ===
        # Bull Power > 0 (close > EMA13) AND ADX > 25 (trending) AND volume confirmation
        if (bull_power_aligned[i] > 0) and (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Bear Power < 0 (close < EMA13) AND ADX > 25 (trending) AND volume confirmation
        elif (bear_power_aligned[i] < 0) and (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dADX_Volume_Filter_v2"
timeframe = "6h"
leverage = 1.0