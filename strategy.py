#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR expansion filter
# Long when price breaks above 12h Donchian high(20) AND 12h volume > 1.5x 20-period volume SMA AND ATR(14) > ATR(50)
# Short when price breaks below 12h Donchian low(20) AND 12h volume > 1.5x 20-period volume SMA AND ATR(14) > ATR(50)
# Uses 12h Donchian channels for structure, volume confirmation for validity, and ATR expansion to avoid chop
# Works in bull (breakouts above upper channel) and bear (breakdowns below lower channel) via symmetric logic
# Discrete sizing 0.25 limits drawdown; targets 20-50 trades/year to avoid fee drag

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
    
    # Get 12h data once before loop for Donchian, volume, and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicator: Donchian Channel (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian high and low (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 12h Indicator: Volume SMA (20-period) for confirmation ===
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 12h Indicator: ATR (14-period and 50-period) for volatility regime filter ===
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_12h, atr_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # Need 50 for ATR, 20 for Donchian and volume SMA, extra buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_sma_20_12h_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 1.5
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Volatility filter: ATR(14) > ATR(50) - ensures we're in expanding volatility regime
        vol_expansion = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === LONG CONDITIONS ===
        # Price breaks above 12h Donchian high(20) AND volume confirmation AND volatility expansion
        if (price > donchian_high_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below 12h Donchian low(20) AND volume confirmation AND volatility expansion
        elif (price < donchian_low_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hVolume1.5x_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0