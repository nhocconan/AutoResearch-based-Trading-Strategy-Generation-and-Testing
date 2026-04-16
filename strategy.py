#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and ATR regime filter
# Long when price breaks above upper Donchian channel AND 1w volume > 2.0x 20-period volume SMA AND ATR(14) > ATR(50)
# Short when price breaks below lower Donchian channel AND 1w volume > 2.0x 20-period volume SMA AND ATR(14) > ATR(50)
# Uses Donchian channels from 1d timeframe for structure, 1w volume confirmation for validity, and ATR expansion filter
# Works in bull (breakouts above upper channel) and bear (breakdowns below lower channel) via symmetric logic
# Discrete sizing 0.25 limits drawdown; targets 20-40 trades/year to avoid fee drag

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
    
    # Get 1d data once before loop for Donchian channels, volume, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Get 1w data once before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper and lower Donchian channels
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # === 1w Indicator: Volume SMA (20-period) for confirmation ===
    volume_1w = df_1w['volume'].values
    vol_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma_20_1w)
    
    # === 1d Indicator: ATR (14-period and 50-period) for volatility regime filter ===
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    close_1d = df_1d['close'].values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 80  # Need 50 for ATR, 20 for Donchian and volume SMA, extra buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_sma_20_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1w volume (aligned)
        vol_1w_series = df_1w['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_series)
        if np.isnan(vol_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1w volume > 2.0x 20-period 1w volume SMA
        vol_threshold = vol_sma_20_1w_aligned[i] * 2.0
        vol_confirm = vol_1w_aligned[i] > vol_threshold
        
        # Volatility filter: ATR(14) > ATR(50) - ensures we're in expanding volatility regime
        vol_expansion = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === LONG CONDITIONS ===
        # Price breaks above upper Donchian channel AND volume confirmation AND volatility expansion
        if (price > upper_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below lower Donchian channel AND volume confirmation AND volatility expansion
        elif (price < lower_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_Volume2.0x_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0