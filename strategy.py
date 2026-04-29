#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout + 12h EMA50 Trend + Volume Spike + Chop Filter
# Long when price breaks above Camarilla R3 level AND price > 12h EMA50 AND volume > 2.0x 20-bar avg AND chop < 61.8 (trending)
# Short when price breaks below Camarilla S3 level AND price < 12h EMA50 AND volume > 2.0x 20-bar avg AND chop < 61.8 (trending)
# Exit when price reverts to Camarilla Pivot level (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 19-50 trades/year on 4h timeframe.
# Combines Camarilla precision breakouts with 12h trend filter, volume confirmation, and chop regime filter to avoid ranging markets.
# This tightens entry conditions vs R1/S1 versions to reduce overtrading while maintaining edge in both bull/bear markets via trend filter.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h data
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
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
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Choppiness Index regime filter (14-period) - avoid ranging markets
    # CHOP > 61.8 = ranging (choppy), CHOP < 38.2 = trending
    # We want trending markets: CHOP < 61.8
    atr_period = 14
    true_high = np.maximum(high, np.roll(close, 1))
    true_low = np.minimum(low, np.roll(close, 1))
    true_high[0] = high[0]
    true_low[0] = low[0]
    tr = np.maximum(true_high - true_low, 
                    np.maximum(true_high - np.roll(close, 1),
                               np.roll(close, 1) - true_low))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_sum = highest_high - lowest_low
    range_sum = np.where(range_sum == 0, 1e-10, range_sum)
    
    chop = 100 * np.log10(atr * atr_period / range_sum) / np.log10(atr_period)
    chop_filter = chop < 61.8  # Trending regime (not choppy/ranging)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14)  # volume MA, EMA50, and chop warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        curr_ema50 = ema_50_12h_aligned[i]
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
            # Long when price breaks above Camarilla R3 AND price > 12h EMA50 AND volume confirmation AND trending regime
            if curr_close > curr_R3 and curr_close > curr_ema50 and vol_conf and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 12h EMA50 AND volume confirmation AND trending regime
            elif curr_close < curr_S3 and curr_close < curr_ema50 and vol_conf and chop_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals