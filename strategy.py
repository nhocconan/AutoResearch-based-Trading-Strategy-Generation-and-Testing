#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ATR regime filter
# Long when price breaks above R3 AND 1d volume > 1.8x 20-period volume SMA AND ATR(14) > ATR(50)
# Short when price breaks below S3 AND 1d volume > 1.8x 20-period volume SMA AND ATR(14) > ATR(50)
# Uses Camarilla pivot levels from 1d timeframe for structure, volume confirmation for validity, and ATR expansion filter
# Works in bull (breakouts above R3) and bear (breakdowns below S3) via symmetric logic
# Discrete sizing 0.25 limits drawdown; targets 20-40 trades/year to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop (for session alignment only - we use 1d for Camarilla)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Get 1d data once before loop for Camarilla pivots, volume, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla width
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0
    # R3 and S3 levels
    r3 = close_1d + camarilla_width * 1.1
    s3 = close_1d - camarilla_width * 1.1
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1d Indicator: ATR (14-period and 50-period) for volatility regime filter ===
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 80  # Need 50 for ATR, 20 for volume SMA, extra buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.8x 20-period 1d volume SMA
        vol_threshold = vol_sma_20_1d_aligned[i] * 1.8
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Volatility filter: ATR(14) > ATR(50) - ensures we're in expanding volatility regime
        vol_expansion = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === LONG CONDITIONS ===
        # Price breaks above Camarilla R3 AND volume confirmation AND volatility expansion
        if (price > r3_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below Camarilla S3 AND volume confirmation AND volatility expansion
        elif (price < s3_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_Volume1.8x_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0