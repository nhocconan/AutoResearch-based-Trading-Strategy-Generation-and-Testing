#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ATR expansion filter
# Long when price breaks above 1d Camarilla R3 level AND 1d volume > 2.0x 20-period volume SMA AND ATR(14) > ATR(50)
# Short when price breaks below 1d Camarilla S3 level AND 1d volume > 2.0x 20-period volume SMA AND ATR(14) > ATR(50)
# Uses 1d Camarilla pivots for structure, volume spike for conviction, and ATR expansion to avoid chop
# Works in bull (breakouts above R3) and bear (breakdowns below S3) via symmetric logic
# Discrete sizing 0.25 limits drawdown; targets 15-30 trades/year to avoid fee drag

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
    
    # Get 1d data once before loop for Camarilla, volume, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3 and S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R3 = close + range * 1.1/4, S3 = close - range * 1.1/4
    camarilla_r3 = close_1d + (range_hl * 1.1 / 4)
    camarilla_s3 = close_1d - (range_hl * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
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
    warmup = 100  # Need 50 for ATR, 20 for volume SMA
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 2.0x 20-period 1d volume SMA
        vol_threshold = vol_sma_20_1d_aligned[i] * 2.0
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Volatility filter: ATR(14) > ATR(50) - ensures we're in expanding volatility regime
        vol_expansion = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === LONG CONDITIONS ===
        # Price breaks above 1d Camarilla R3 AND volume confirmation AND volatility expansion
        if (price > camarilla_r3_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below 1d Camarilla S3 AND volume confirmation AND volatility expansion
        elif (price < camarilla_s3_aligned[i]) and vol_confirm and vol_expansion:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolume2x_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0