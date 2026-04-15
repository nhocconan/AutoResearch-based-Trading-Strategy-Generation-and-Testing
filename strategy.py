#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above 4h BB upper (20,2) + 1d ATR(14) > 30-period SMA(ATR) + volume > 1.5x 20-period avg
# Short when price breaks below 4h BB lower (20,2) + 1d ATR(14) > 30-period SMA(ATR) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years.
# Bollinger Bands provide volatility-adjusted breakout levels. ATR regime filter ensures we only trade during
# elevated volatility periods, avoiding low-volatility chop. Works in bull markets (trend continuation) and
# bear markets (strong downtrends) by requiring volatility expansion. Session filter (08-20 UTC) reduces noise.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # === 1d Indicator: ATR Regime Filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    period = 14
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # ATR regime: current ATR > 30-period SMA of ATR
    atr_sma_30 = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
    atr_regime = atr > atr_sma_30
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # === 4h Indicators: Bollinger Bands (20,2) ===
    bb_window = 20
    bb_std = 2
    
    # Calculate BB using 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < bb_window:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Middle Band (SMA)
    bb_middle = pd.Series(close_4h).rolling(window=bb_window, min_periods=bb_window).mean().values
    # Standard Deviation
    bb_std_dev = pd.Series(close_4h).rolling(window=bb_window, min_periods=bb_window).std().values
    # Upper and Lower Bands
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # Align BB to 1h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 30) + 20  # BB(20) + ATR_SMA(30) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h BB upper (20,2)
        # 2. Volatility regime (1d ATR > 30-period SMA of ATR)
        # 3. Volume confirmation
        if (close[i] > bb_upper_aligned[i]) and \
           (atr_regime_aligned[i] > 0.5) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h BB lower (20,2)
        # 2. Volatility regime (1d ATR > 30-period SMA of ATR)
        # 3. Volume confirmation
        elif (close[i] < bb_lower_aligned[i]) and \
             (atr_regime_aligned[i] > 0.5) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_BB20_1dATR_Regime_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0