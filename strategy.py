#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation
# Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period volume SMA
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d ADX > 25 AND volume > 1.5x 20-period volume SMA
# Uses 1d ADX for regime filter (only trade in trending markets) and volume confirmation for validity
# Works in bull (strong Bull Power) and bear (strong Bear Power) via symmetric logic
# Discrete sizing 0.25 targets 12-30 trades/year to avoid fee drag

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
    
    # Get 1d data once before loop for Elder Ray, ADX, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: EMA(13) for Elder Ray ===
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 1d Indicator: Elder Ray Index ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13
    bear_power = ema_13 - low_1d
    
    # === 1d Indicator: ADX (14-period) for regime filter ===
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)  # for reference
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # Need 13+14+14 for EMA/ADX, 20 for volume SMA, extra buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or 
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Regime filter: ADX > 25 (trending market)
        trending = adx_aligned[i] > 25.0
        
        # Price levels
        price = close[i]
        
        # === LONG CONDITIONS ===
        # Bull Power > 0 (strong buying pressure) AND Bear Power rising (less negative) AND trending AND volume confirmation
        bull_rising = bear_power_aligned[i] > bear_power_aligned[i-1]  # Bear Power becoming less negative
        if (bull_power_aligned[i] > 0) and bull_rising and trending and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Bear Power < 0 (strong selling pressure) AND Bull Power falling (less positive) AND trending AND volume confirmation
        bull_falling = bull_power_aligned[i] < bull_power_aligned[i-1]  # Bull Power becoming less positive
        if (bear_power_aligned[i] < 0) and bull_falling and trending and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dADX_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0