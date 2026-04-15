#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike filter and ATR-based stoploss
# Long when price breaks above Camarilla R3 (1d) + volume > 2.0x 20-period avg
# Short when price breaks below Camarilla S3 (1d) + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-60 trades/year.
# Camarilla levels provide intraday support/resistance. Volume spike confirms institutional interest.
# Works in bull markets (breakouts continue) and bear markets (breakdowns accelerate) with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4.0)  # R3 = pivot + (range * 1.1/4)
    s3 = pivot - (range_1d * 1.1 / 4.0)  # S3 = pivot - (range * 1.1/4)
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 4h Indicators: Volume SMA (20-period) and ATR (14-period) for stoploss ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, atr_period) + 5  # Volume(20) + ATR(14) + buffer
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation
        if (close[i] > r3_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation
        elif (close[i] < s3_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_1dVolumeSpike_v1"
timeframe = "4h"
leverage = 1.0