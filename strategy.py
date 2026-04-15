#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above 12h Donchian upper + 1d ATR(14) < 30-period SMA(ATR) (low vol regime) + volume > 1.5x 20-period avg
# Short when price breaks below 12h Donchian lower + 1d ATR(14) < 30-period SMA(ATR) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1d ATR regime filter identifies low volatility environments where breakouts are more likely to trend.
# Volume threshold (1.5x) targets ~20-40 trades/year to minimize fee drag on 12h timeframe.
# Donchian channels calculated from 12h data using 20-period lookback.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ATR(14) and its 30-period SMA for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 30-period SMA of ATR
    atr_sma_30_1d = pd.Series(atr_14_1d).rolling(window=30, min_periods=30).mean().values
    
    # ATR regime: low volatility when current ATR < SMA of ATR
    atr_regime_low_vol = atr_14_1d < atr_sma_30_1d
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_low_vol.astype(float))
    
    # === 12h Donchian Channels (20-period) ===
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 5  # ATR(14)+SMA(30) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # ATR regime filter: low volatility environment
        vol_regime = bool(atr_regime_aligned[i])
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper band)
        # 2. Low volatility regime (ATR < ATR SMA)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and vol_regime and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower band)
        # 2. Low volatility regime (ATR < ATR SMA)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and vol_regime and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATRRegime_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0