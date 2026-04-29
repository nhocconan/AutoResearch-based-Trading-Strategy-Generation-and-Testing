#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
# Long when price breaks above Camarilla R3 level AND price > 1w EMA50 AND volume > 2.0x 20-bar avg AND chop < 61.8
# Short when price breaks below Camarilla S3 level AND price < 1w EMA50 AND volume > 2.0x 20-bar avg AND chop < 61.8
# Exit when price reverts to Camarilla Pivot level (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Camarilla R3/S3 are stronger breakout levels than R1/S1, reducing false breakouts.
# 1w EMA50 filter ensures we only trade with the primary trend.
# Volume confirmation ensures breakout strength.
# Chop filter (choppiness index > 61.8 = ranging) avoids whipsaws in sideways markets.
# Designed to work in both bull and bear markets via trend filter and mean-reversion exits.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_ChopFilter_v1"
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
    
    # Get 1w data for EMA50 trend filter and 1d data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w data
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle first bar where shift creates NaN
    if len(prev_high) > 0:
        prev_high[0] = df_1d['high'].iloc[0]
        prev_low[0] = df_1d['low'].iloc[0]
        prev_close[0] = df_1d['close'].iloc[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + camarilla_range * 1.1 / 4.0  # R3 = C + (H-L)*1.1/4
    camarilla_S3 = prev_close - camarilla_range * 1.1 / 4.0  # S3 = C - (H-L)*1.1/4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Choppiness Index filter: chop < 61.8 = trending (avoid ranging markets)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high)-min(low)))) / log10(n)
    # We'll use a simplified version: ATR(14) / (HHV(14)-LLV(14)) < 0.618
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_series.rolling(window=14, min_periods=14).max().values
    ll_14 = low_series.rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    
    # Chop value: (ATR(14) / range_14) * 100, but we compare directly to 0.618
    chop_value = np.divide(atr_14, range_14, out=np.full_like(atr_14, np.nan), where=range_14!=0)
    chop_filter = chop_value < 0.618  # chop < 61.8 = trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14)  # volume MA, EMA50, and chop warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        curr_ema50 = ema_50_1w_aligned[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_close = close[i]
        
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
            # Long when price breaks above Camarilla R3 AND price > 1w EMA50 AND volume confirmation AND chop filter
            if curr_close > curr_R3 and curr_close > curr_ema50 and vol_conf and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1w EMA50 AND volume confirmation AND chop filter
            elif curr_close < curr_S3 and curr_close < curr_ema50 and vol_conf and chop_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals