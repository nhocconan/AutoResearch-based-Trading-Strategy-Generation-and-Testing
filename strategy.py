#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1d EMA50 Trend + Volume Spike + Choppiness Filter
# Long when price breaks above Camarilla R3 level AND price > 1d EMA50 AND volume > 2.0x 20-bar avg AND chop < 61.8
# Short when price breaks below Camarilla S3 level AND price < 1d EMA50 AND volume > 2.0x 20-bar avg AND chop < 61.8
# Exit when price reverts to Camarilla Pivot level (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Choppiness filter avoids whipsaws in ranging markets. Camarilla R3/S3 provides stronger breakout signals.
# Designed to work in both bull and bear markets via trend filter and mean-reversion exits.

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla pivot calculation, EMA50 trend filter, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Handle first bar where shift creates NaN
    if len(prev_high) > 0:
        prev_high[0] = df_1d['high'].iloc[0]
        prev_low[0] = df_1d['low'].iloc[0]
        prev_close[0] = df_1d['close'].iloc[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + camarilla_range * 1.1 / 4.0
    camarilla_S3 = prev_close - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Choppiness Index: CHOP = 100 * log10(SUM(ATR(1)) / (MAX(HIGH)-MIN(LOW))) / log10(n)
    # Calculate ATR(1) for 1d data
    tr1 = np.maximum(df_1d['high'].values, df_1d['close'].shift(1).values) - np.minimum(df_1d['low'].values, df_1d['close'].shift(1).values)
    tr1[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    
    # Calculate CHOP over 14 periods
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    chop = 100 * np.log10(atr_sum / range_14) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop), 50.0, chop)
    # Align CHOP to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14)  # volume MA, EMA50, and CHOP warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_close = close[i]
        curr_chop = chop_aligned[i]
        
        # Choppiness filter: only trade when market is trending (CHOP < 61.8)
        chop_filter = curr_chop < 61.8
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 1d EMA50 AND volume confirmation AND chop filter
            if curr_close > curr_R3 and curr_close > curr_ema50 and vol_conf and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1d EMA50 AND volume confirmation AND chop filter
            elif curr_close < curr_S3 and curr_close < curr_ema50 and vol_conf and chop_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals