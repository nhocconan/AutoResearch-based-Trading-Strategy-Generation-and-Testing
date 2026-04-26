#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_ChopFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 4h EMA50 trend filter, volume confirmation, and chop regime filter.
Long when price breaks above R1 in 4h uptrend with volume spike and chop<61.8 (trending).
Short when price breaks below S1 in 4h downtrend with volume spike and chop<61.8.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year on 4h.
Works in bull/bear by following 4h trend. Camarilla levels provide intraday support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation, EMA trend, and chop filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # need 14 for chop + 20 for EMA + 1 for Camarilla
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h (based on previous day's OHLC)
    # Camarilla uses previous period's OHLC to calculate support/resistance for current period
    # For 4h timeframe, we use previous 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's OHLC (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    # Set first value to NaN since we don't have previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R1, S1 levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend_4h = close_4h > ema_50_4h
    downtrend_4h = close_4h < ema_50_4h
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_4h * 1.5)
    
    # Choppiness Index filter (CHOP < 61.8 = trending)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # We'll use a simplified version: CHOP < 61.8 indicates trending market
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate CHOP: 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(atr_period)
    chop_filter = chop < 61.8  # trending regime
    
    # Align all 4h indicators to 6h timeframe (wait for completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike.astype(float))
    chop_filter_aligned = align_htf_to_ltf(prices, df_4h, chop_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for ATR-based CHOP, 20 for volume MA, 50 for EMA)
    start_idx = max(30, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(uptrend_4h_aligned[i]) or
            np.isnan(downtrend_4h_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_filter_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend, volume spike, and trending regime
            if (close[i] > r1_aligned[i] and 
                uptrend_4h_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 4h downtrend, volume spike, and trending regime
            elif (close[i] < s1_aligned[i] and 
                  downtrend_4h_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 OR 4h trend changes to downtrend
            if (close[i] < s1_aligned[i] or not uptrend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 OR 4h trend changes to uptrend
            if (close[i] > r1_aligned[i] or not downtrend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0