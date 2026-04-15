#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above 12h Donchian upper (20-period high) + 1d ATR(14) < median ATR + volume > 1.5x 20-period avg
# Short when price breaks below 12h Donchian lower (20-period low) + 1d ATR(14) < median ATR + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1d ATR filter avoids high volatility regimes where breakouts fail, targeting low-volatility expansion moves.
# Volume confirmation ensures breakouts have participation. Target: ~12-25 trades/year to minimize fee drag on 12h.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: ATR(14) and its median ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Median ATR over 50 periods for regime filter
    median_atr_50 = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).median().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    median_atr_50_aligned = align_htf_to_ltf(prices, df_1d, median_atr_50)
    
    # === 12h Donchian Channels (20-period) ===
    # Upper = 20-period high, Lower = 20-period low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # ATR(14)+median(50) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(median_atr_50_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Volatility regime filter: current ATR < median ATR (low volatility environment)
        vol_regime = atr_14_1d_aligned[i] < median_atr_50_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. Low volatility regime (ATR < median ATR)
        # 3. Volume confirmation
        if (close[i] > donch_high[i]) and vol_regime and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. Low volatility regime (ATR < median ATR)
        # 3. Volume confirmation
        elif (close[i] < donch_low[i]) and vol_regime and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATR_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0