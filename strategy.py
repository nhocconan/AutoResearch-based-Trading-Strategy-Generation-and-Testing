#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above Donchian(20) high + 1d ATR(14) < 30-period median (low volatility) + volume > 1.5x 20-period avg
# Short when price breaks below Donchian(20) low + 1d ATR(14) < 30-period median + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1d ATR regime filter avoids choppy markets where breakouts fail, improving win rate in both bull and bear.
# Volume confirmation ensures breakouts have participation, reducing false signals.
# Donchian(20) provides clear structure-based entries with proven edge on 4h/12h timeframes.

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
    
    # === 1d Indicator: ATR(14) and its 30-period median for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # 30-period median of ATR
    atr_median_30_1d = pd.Series(atr_14_1d).rolling(window=30, min_periods=30).median().values
    # Regime: low volatility when ATR < median
    low_vol_regime = atr_14_1d < atr_median_30_1d
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # === 12h Donchian(20) channels (based on prior 20 bars) ===
    # Donchian High = max(high of prior 20 bars)
    # Donchian Low = min(low of prior 20 bars)
    # Using rolling window on prior bar data to avoid look-ahead
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 30, 20) + 5  # Donchian(20) + ATR median(30) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(low_vol_regime_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Regime filter: low volatility environment (ATR < median)
        regime_filter = low_vol_regime_aligned[i] > 0.5
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian(20) high (close > Donchian High)
        # 2. Low volatility regime (avoid choppy markets)
        # 3. Volume confirmation
        if (close[i] > donchian_h[i]) and \
           regime_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian(20) low (close < Donchian Low)
        # 2. Low volatility regime (avoid choppy markets)
        # 3. Volume confirmation
        elif (close[i] < donchian_l[i]) and \
             regime_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATR_Regime_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0