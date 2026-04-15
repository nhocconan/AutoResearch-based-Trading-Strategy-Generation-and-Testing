#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volume filter and session time filter
# Long when price breaks above Donchian upper band (20-period high) + 1d ATR(14) > 1.5x 50-period median ATR + volume > 1.5x 20-period volume average
# Short when price breaks below Donchian lower band (20-period low) + 1d ATR(14) > 1.5x 50-period median ATR + volume > 1.5x 20-period volume average
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# ATR-based volatility filter ensures we only trade during sufficient volatility regimes, reducing whipsaws.
# Volume confirmation (1.5x) targets ~25-35 trades/year on 12h timeframe to avoid overtrading.
# Session filter (08-20 UTC) focuses on active trading hours.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: ATR(14) and its 50-period median for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = 0  # no previous close for first period
    tr3[0] = 0  # no previous close for first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[13] = np.mean(tr[0:14])  # seed with simple average
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # 50-period median of ATR for regime filter
    atr_series = pd.Series(atr_14_1d)
    atr_median_50 = atr_series.rolling(window=50, min_periods=50).median().values
    
    # Volatility regime: ATR(14) > 1.5x 50-period median ATR
    vol_regime = atr_14_1d > (atr_median_50 * 1.5)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # === 12h Donchian Channel (20-period) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(60, 20) + 5  # ATR(14)+50median + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Volatility regime filter: must be in high volatility regime
        if not vol_regime_aligned[i]:
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper band (close > upper)
        # 2. Volume confirmation
        # 3. Volatility regime filter
        if (close[i] > donchian_upper[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower band (close < lower)
        # 2. Volume confirmation
        # 3. Volatility regime filter
        elif (close[i] < donchian_lower[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATR_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0