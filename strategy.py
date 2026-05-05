#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike + Chop Regime
# Long when price breaks above R3 (1d) AND price > 1d EMA34 (strong uptrend) AND volume spike AND choppy regime (mean reversion favor)
# Short when price breaks below S3 (1d) AND price < 1d EMA34 (strong downtrend) AND volume spike AND choppy regime
# Uses 1d EMA34 for smoother trend filter than shorter EMAs, reducing whipsaw
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Chop regime filter (CHOP > 61.8) adds mean-reversion bias in ranging markets, improving bear market performance
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag while capturing high-probability breaks
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation in chop)
# Timeframe: 4h (primary timeframe as required)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use only completed daily bar (look-ahead safety)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    
    # Calculate pivot point (PP) = (H+L+C)/3
    pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3.0
    # Calculate range
    range_1d = high_1d_shifted - low_1d_shifted
    # Camarilla levels (R3/S3 = PP ± range*1.1/2)
    r3 = pp + (range_1d * 1.1 / 2.0)  # R3 = PP + range*1.1/2
    s3 = pp - (range_1d * 1.1 / 2.0)  # S3 = PP - range*1.1/2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation on 4h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Chop regime filter on 1d (CHOP > 61.8 = choppy/ranging market)
    if len(df_1d) >= 14:
        high_1d_roll = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        low_1d_roll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        atr_14_1d = pd.Series(high_1d_roll - low_1d_roll).ewm(span=14, adjust=False, min_periods=14).mean().values
        sum_abs_diff = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=14, min_periods=14).sum().values
        chop = 100 * np.log10(sum_abs_diff / (atr_14_1d * 14)) / np.log10(10)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
        chop_filter = chop_aligned > 61.8  # Choppy regime favors mean reversion
    else:
        chop_filter = np.ones(n, dtype=bool)  # Default to true if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND strong uptrend (price > 1d EMA34) AND volume spike AND choppy regime
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND strong downtrend (price < 1d EMA34) AND volume spike AND choppy regime
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 OR closes below 1d EMA34
            if close[i] < r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S3 OR closes above 1d EMA34
            if close[i] > s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals