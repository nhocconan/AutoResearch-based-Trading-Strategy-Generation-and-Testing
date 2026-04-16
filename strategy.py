#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla pivot levels (S3/R3) from 1d timeframe
# with volume spike confirmation and choppiness regime filter.
# Long when price breaks above R3 with volume > 1.5x average and chop > 61.8 (ranging).
# Short when price breaks below S3 with volume > 1.5x average and chop > 61.8 (ranging).
# Exit when price reverts to pivot point (PP) or chop < 38.2 (trending).
# Uses discrete position size 0.25. Camarilla levels provide intraday support/resistance
# that work well in ranging markets. Volume spike confirms breakout legitimacy.
# Choppiness filter ensures we only trade in ranging regimes where mean reversion works.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to balance opportunity
# and cost efficiency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (S3, S2, S1, PP, R1, R2, R3) ===
    # PP = (High + Low + Close) / 3
    # R3 = PP + (High - Low) * 1.1
    # S3 = PP - (High - Low) * 1.1
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1
    
    # Align Camarilla levels to primary timeframe (4h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # === 4h Indicators: Volume Spike and Choppiness Index ===
    # Volume ratio: current volume / 20-period average volume
    vol_sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values  # using close as proxy for volume SMA
    vol_ratio = volume / np.where(vol_sma > 0, vol_sma, 1.0)  # avoid division by zero
    
    # Choppiness Index: measures whether market is ranging (high values) or trending (low values)
    # CHOP = 100 * log10(sum(ATR over n periods) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    chop = 100 * np.log10(atr_sum / np.where(range_hl > 0, range_hl, 1.0)) / np.log10(14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # covers 20-period volume SMA and 14-period chop
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        pp = pp_aligned[i]
        vol_ratio_current = vol_ratio[i]
        chop_current = chop[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reverts to pivot point OR market starts trending (chop < 38.2)
            if (price <= pp) or (chop_current < 38.2):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reverts to pivot point OR market starts trending (chop < 38.2)
            if (price >= pp) or (chop_current < 38.2):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 with volume spike AND chop indicates ranging market
            if (price > r3) and (vol_ratio_current > 1.5) and (chop_current > 61.8):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S3 with volume spike AND chop indicates ranging market
            elif (price < s3) and (vol_ratio_current > 1.5) and (chop_current > 61.8):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dCamarilla_S3R3_VolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0