#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ATR-based volatility filter and volume spike
# Long when: price breaks above R3, volume > 2x 20-period average, and 1d ATR(14) > 1.5x ATR(50) (expanding volatility)
# Short when: price breaks below S3, volume > 2x 20-period average, and 1d ATR(14) > 1.5x ATR(50)
# Exit when price returns to Camarilla R3/S3 level (mean reversion) or opposite breakout
# Uses volatility expansion to capture breakouts in both bull and bear markets, reducing false signals in choppy conditions.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3S3_Breakout_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ATR and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility expansion filter
    if len(high_1d) >= 14:
        # True Range calculation
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr1[0] = high_1d[0] - low_1d[0]  # First TR
        tr2[0] = np.abs(high_1d[0] - close_1d[0])
        tr3[0] = np.abs(low_1d[0] - close_1d[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
        # Volatility expansion: short-term ATR > 1.5x long-term ATR
        vol_expansion = atr_14 > (1.5 * atr_50)
    else:
        atr_14 = np.full(len(close_1d), np.nan)
        atr_50 = np.full(len(close_1d), np.nan)
        vol_expansion = np.zeros(len(close_1d), dtype=bool)
    
    # Calculate Camarilla levels from previous 1d bar
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * rang * 1.1 / 4
        camarilla_s3 = prev_close - 1.1 * rang * 1.1 / 4
    else:
        camarilla_r3 = np.full(len(close_1d), np.nan)
        camarilla_s3 = np.full(len(close_1d), np.nan)
    
    # Align HTF indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) if 'ema_50_1d_aligned' in locals() else False) or \
           np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(volume_filter[i]) or \
           np.isnan(vol_expansion_aligned[i]) if hasattr(vol_expansion_aligned, '__len__') else False:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Handle vol_expansion_aligned which might be boolean array
        vol_exp = vol_expansion_aligned[i] if hasattr(vol_expansion_aligned, '__getitem__') else vol_expansion_aligned
        
        if position == 0:
            # Long conditions: price breaks above R3, volume filter, and volatility expansion
            if (close[i] > camarilla_r3_aligned[i] and 
                open_price[i] <= camarilla_r3_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                vol_exp):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3, volume filter, and volatility expansion
            elif (close[i] < camarilla_s3_aligned[i] and 
                  open_price[i] >= camarilla_s3_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  vol_exp):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 (mean reversion) or breaks below S3 (reversal)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 (mean reversion) or breaks above R3 (reversal)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals