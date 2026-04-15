#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# Long when price breaks above Donchian upper band + 1d ATR(14) > 1.5x its 50-period SMA (high volatility regime) + volume > 1.5x 20-period avg
# Short when price breaks below Donchian lower band + same volatility + volume filter
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Volatility filter ensures we only trade during explosive moves, reducing whipsaws in ranging markets.
# Volume confirmation (1.5x) targets ~25-35 trades/year on 12h timeframe to avoid overtrading.
# Donchian channels provide structure-based breakout levels that work in both bull and bear markets.

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
    
    # === 1d Indicator: ATR(14) and its 50-period SMA for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    volatility_filter = align_htf_to_ltf(prices, df_1d, atr_14_1d > (atr_ma_50_1d * 1.5))
    
    # === 12h Donchian Channel (20-period) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(60, 20) + 5  # ATR(14) + MA(50) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Volatility filter: 1d ATR(14) > 1.5x its 50-period SMA
        vol_regime = bool(volatility_filter[i])
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper band (close > upper)
        # 2. High volatility regime (1d ATR > 1.5x MA)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and vol_regime and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower band (close < lower)
        # 2. High volatility regime (1d ATR > 1.5x MA)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and vol_regime and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATR_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0