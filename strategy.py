#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above upper BB(20,2) + ATR(1d) > ATR_SMA(1d,50) + volume > 1.5x volume SMA(20)
# Short when price breaks below lower BB(20,2) + ATR(1d) > ATR_SMA(1d,50) + volume > 1.5x volume SMA(20)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# Bollinger Bands provide dynamic support/resistance that adapts to volatility.
# ATR regime filter ensures we only trade when volatility is expanding (above average), avoiding low-volatility chop.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by requiring expanding volatility.

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
    
    # === 1d Indicator: ATR regime filter (ATR > ATR_SMA(50)) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate ATR_SMA(50)
    atr_sma_period = 50
    atr_sma = np.zeros_like(atr)
    for i in range(atr_sma_period-1, len(atr)):
        atr_sma[i] = np.mean(atr[i-atr_sma_period+1:i+1])
    
    # ATR regime: 1 when ATR > ATR_SMA(50) (expanding volatility), 0 otherwise
    atr_regime = (atr > atr_sma).astype(float)
    
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # === 4h Indicator: Bollinger Bands (20,2) ===
    bb_period = 20
    bb_std = 2
    
    # Calculate SMA
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    
    # Calculate standard deviation
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Upper and lower bands
    bb_upper = sma + (bb_stddev * bb_std)
    bb_lower = sma - (bb_stddev * bb_std)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_period, atr_period + atr_sma_period) + 20  # BB(20) + ATR(14)+SMA(50) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Bollinger upper band (20,2)
        # 2. Volatility regime: ATR(1d) > ATR_SMA(1d,50) (expanding volatility)
        # 3. Volume confirmation
        if (close[i] > bb_upper[i]) and \
           (atr_regime_aligned[i] > 0.5) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Bollinger lower band (20,2)
        # 2. Volatility regime: ATR(1d) > ATR_SMA(1d,50) (expanding volatility)
        # 3. Volume confirmation
        elif (close[i] < bb_lower[i]) and \
             (atr_regime_aligned[i] > 0.5) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_BB20_2_ATRRegime1d_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0