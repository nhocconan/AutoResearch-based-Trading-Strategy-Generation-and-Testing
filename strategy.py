#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ATR-based volatility filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) + 12h ATR ratio (current/20-period) > 1.2 + volume > 1.3x 20-period avg
# Short when price breaks below 4h Donchian lower (20-period) + 12h ATR ratio > 1.2 + volume > 1.3x 20-period avg
# Uses ATR ratio to filter for expanding volatility (avoids chop) and volume confirmation for conviction.
# Works in bull markets (breakouts with momentum) and bear markets (breakdowns with panic selling) by requiring volatility expansion.
# Discrete position sizing (0.25) minimizes fee churn. Target trade frequency: 20-40/year.

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: ATR and ATR ratio (volatility filter) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    high_12h_shift = np.roll(high_12h, 1)
    low_12h_shift = np.roll(low_12h, 1)
    close_12h_shift = np.roll(close_12h, 1)
    high_12h_shift[0] = high_12h[0]
    low_12h_shift[0] = low_12h[0]
    close_12h_shift[0] = close_12h[0]
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - close_12h_shift)
    tr3 = np.abs(low_12h - close_12h_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period) using Wilder's smoothing
    period = 14
    atr_12h = np.zeros_like(tr)
    atr_12h[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_12h[i] = (atr_12h[i-1] * (period-1) + tr[i]) / period
    
    # ATR ratio: current ATR / 20-period ATR SMA (expanding volatility filter)
    atr_sma_20 = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_sma_20 > 0, atr_12h / atr_sma_20, 1.0)
    
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 20) + 20  # Donchian(20) + ATR(14)+SMA(20) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Volatility expansion (12h ATR ratio > 1.2)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           (atr_ratio_aligned[i] > 1.2) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Volatility expansion (12h ATR ratio > 1.2)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             (atr_ratio_aligned[i] > 1.2) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hATRratio_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0