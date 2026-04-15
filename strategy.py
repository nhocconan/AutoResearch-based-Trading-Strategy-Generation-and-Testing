#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and ATR-based trend filter.
# In low volatility regimes (ATR contraction), breakouts are more likely to fail → require volume spike.
# In high volatility regimes (ATR expansion), breakouts have higher follow-through → volume filter relaxed.
# Uses 12h EMA50 for multi-timeframe trend alignment to avoid counter-trend trades.
# Designed for low trade frequency (20-40/year) with discrete position sizing (0.25) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicators: EMA(50) for trend filter ===
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Calculate rolling max/min for Donchian channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR(14) for volatility regime ===
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 20-period ATR SMA (volatility regime filter)
    atr_sma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_sma_20  # >1 = expanding vol, <1 = contracting vol
    
    # Volume confirmation: current volume > 1.5x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20 * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY REGIME ADJUSTMENT ===
        # In contracting volatility (ATR ratio < 0.8): require stronger volume confirmation
        # In expanding volatility (ATR ratio > 1.2): relax volume requirement
        # In neutral: standard volume confirmation
        if atr_ratio[i] < 0.8:
            vol_req = vol_confirm[i] and (volume[i] > (vol_sma_20[i] * 2.0))  # 2x volume spike
        elif atr_ratio[i] > 1.2:
            vol_req = vol_confirm[i] or (volume[i] > (vol_sma_20[i] * 1.2))  # relaxed
        else:
            vol_req = vol_confirm[i]  # standard
        
        # === LONG CONDITIONS ===
        # Donchian breakout above 20-period high + volume confirmation + 12h uptrend
        if (close[i] > high_max_20[i]) and vol_req and (close[i] > ema_50_12h_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Donchian breakdown below 20-period low + volume confirmation + 12h downtrend
        elif (close[i] < low_min_20[i]) and vol_req and (close[i] < ema_50_12h_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_VolATR_Regime_EMA50_12h_v1"
timeframe = "4h"
leverage = 1.0