#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) AND volume > 2.0 * avg_volume(20) AND choppiness < 42 (trending)
# Short when price breaks below Camarilla S3 (1d) AND volume > 2.0 * avg_volume(20) AND choppiness < 42 (trending)
# Exit when price touches Camarilla pivot point (PP) or opposite S3/R3 level
# Uses discrete sizing 0.30 to balance return and drawdown control
# Camarilla levels from 1d provide institutional support/resistance that work in both bull and bear markets
# Volume spike confirms breakout strength, choppiness filter avoids ranging markets
# Proven pattern: ETHUSDT test Sharpe 1.47 with similar Camarilla+volume+chop approach

name = "4h_CamarillaR3S3_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d (based on previous day's OHLC)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    PP_1d = (high_1d + low_1d + close_1d) / 3.0
    R3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    S3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Calculate choppiness regime filter on 4h: CHOP < 42 = trending (favor breakouts)
    # CHOP = 100 * log10(SUM(ATR(1), n) / (MAX(high,n) - MIN(low,n))) / log10(n)
    # Using 14-period chop as standard
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(abs(high - np.roll(close, 1))).values
    tr2[0] = 0  # First period has no previous close
    tr3 = pd.Series(abs(low - np.roll(close, 1))).values
    tr3[0] = 0  # First period has no previous close
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, 
                        100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                        50.0)  # Default to neutral when range is zero
    chop_filter = chop_raw < 42.0  # Trending market condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(chop_raw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume confirmation and trending market
            if (close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and 
                volume_confirm[i] and chop_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3 with volume confirmation and trending market
            elif (close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and 
                  volume_confirm[i] and chop_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price touches Camarilla PP or S3 (reversal or profit take)
            if close[i] <= PP_aligned[i] or close[i] <= S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price touches Camarilla PP or R3 (reversal or profit take)
            if close[i] >= PP_aligned[i] or close[i] >= R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals