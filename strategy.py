#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Breakout with 1d ATR Regime Filter and Volume Spike
# Long when price breaks above 6h BB upper (20,2.0) + 1d ATR(14) > 1.5x 50-period MA ATR + volume > 2.0x 20-period avg volume
# Short when price breaks below 6h BB lower (20,2.0) + same regime + volume filter
# Uses Bollinger Bands for volatility-based breakouts. ATR regime ensures we only trade when volatility is expanding (avoiding low-vol chop).
# Volume spike confirms institutional interest. Designed for low trade frequency (15-30/year) to minimize fee drag.
# Works in bull markets (breakouts continue) and bear markets (volatility expansions on downside) by requiring expanding volatility regime.

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
    
    # === 1d Indicator: ATR(14) for volatility regime ===
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
    
    # ATR calculation (Wilder's smoothing)
    atr_period = 14
    alpha = 1.0 / atr_period
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # ATR moving average (50-period) for regime comparison
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR > 1.5x its 50-period MA (expanding volatility)
    vol_regime = atr > (atr_ma * 1.5)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # === 6h Indicator: Bollinger Bands (20,2.0) ===
    bb_window = 20
    bb_std = 2.0
    
    # Basis (SMA)
    basis = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    # Standard deviation
    bb_std_dev = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    # Upper and lower bands
    upper_band = basis + (bb_std_dev * bb_std)
    lower_band = basis - (bb_std_dev * bb_std)
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 50) + 20  # BB(20) + ATR MA(50) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA (strong spike)
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_regime_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h Bollinger Band upper (20,2.0)
        # 2. Volatility regime: 1d ATR > 1.5x 50-period MA ATR (expanding volatility)
        # 3. Volume confirmation (>2.0x average volume)
        if (close[i] > upper_band[i]) and \
           vol_regime_aligned[i] and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h Bollinger Band lower (20,2.0)
        # 2. Volatility regime: 1d ATR > 1.5x 50-period MA ATR (expanding volatility)
        # 3. Volume confirmation (>2.0x average volume)
        elif (close[i] < lower_band[i]) and \
             vol_regime_aligned[i] and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_BB20_2.0_1dATR_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0