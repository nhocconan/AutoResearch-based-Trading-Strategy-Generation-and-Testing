#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume spike filter
# Long when price breaks above 20-period high AND 1d volume > 1.5x 20-period volume SMA AND ATR(14) > ATR(50)
# Short when price breaks below 20-period low AND 1d volume > 1.5x 20-period volume SMA AND ATR(14) > ATR(50)
# Uses volatility expansion (ATR ratio) to filter for genuine breakouts, reducing false signals
# Discrete sizing 0.25 limits drawdown; targets 15-30 trades/year to avoid fee drag
# Works in bull (breakouts continuation) and bear (breakdowns) via symmetric long/short logic

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Get 1d data once before loop for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over 20 periods
    upper_band = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_band = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    upper_band_aligned = align_htf_to_ltf(prices, df_4h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_4h, lower_band)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1d Indicator: ATR (14-period and 50-period) for volatility regime filter ===
    # True Range calculation
    tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
    tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Handle first value where shift creates NaN
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 80  # Need 50 for ATR, 20 for Donchian/volume SMA, extra buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Volatility filter: ATR(14) > ATR(50) - ensures we're in expanding volatility regime
        vol_expansion = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === LONG CONDITIONS ===
        # Price breaks above upper Donchian band AND volume confirmation AND volatility expansion
        if (price > upper_band_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below lower Donchian band AND volume confirmation AND volatility expansion
        elif (price < lower_band_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_VolumeVolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0