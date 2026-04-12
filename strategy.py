#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1d Volume Spike + 1w Choppiness Filter
    # Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend direction and strength
    # Only trade when Alligator is "sleeping" (lines intertwined) then "awakens" (lines diverge)
    # Volume confirmation: current volume > 2.0 * 20-period average to filter weak breakouts
    # Choppiness regime filter: only trade when CHOP(14) < 38.2 (trending) or > 61.8 (ranging)
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Williams Alligator on 12h
    # JAW: 13-period SMMA, shifted 8 bars
    # TEETH: 8-period SMMA, shifted 5 bars  
    # LIPS: 5-period SMMA, shifted 3 bars
    def smma(source, length):
        result = np.full_like(source, np.nan)
        if len(source) < length:
            return result
        # First value is simple average
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CLOSE) / LENGTH
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align to LTF (12h -> 1h alignment needed for 12h timeframe)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_shifted)
    
    # Get 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for choppiness filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and ATR for Choppiness
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])  # align with close_1w
    
    atr_1w = np.full_like(close_1w, np.nan)
    for i in range(14, len(atr_1w)):
        atr_1w[i] = np.mean(tr_1w[i-13:i+1])
    
    # Calculate Choppiness Index: CHOP = 100 * log10(SUM(ATR)/ (HHV - LLV)) / log10(N)
    sum_atr_1w = np.full_like(close_1w, np.nan)
    for i in range(14, len(sum_atr_1w)):
        sum_atr_1w[i] = np.sum(atr_1w[i-13:i+1])
    
    hhvl_1w = np.full_like(close_1w, np.nan)
    llvl_1w = np.full_like(close_1w, np.nan)
    for i in range(14, len(hhvl_1w)):
        hhvl_1w[i] = np.max(high_1w[i-13:i+1])
        llvl_1w[i] = np.min(low_1w[i-13:i+1])
    
    chop_1w = np.full_like(close_1w, np.nan)
    for i in range(14, len(chop_1w)):
        if hhvl_1w[i] != llvl_1w[i]:
            chop_1w[i] = 100 * np.log10(sum_atr_1w[i] / (hhvl_1w[i] - llvl_1w[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Determine Alligator state: sleeping (lines close) or awakening (lines separated)
    # Alligator sleeping: max distance between lines < 0.5% of price
    # Alligator awakening: max distance between lines > 1.5% of price
    max_diff = np.maximum(np.maximum(
        np.abs(jaw_aligned - teeth_aligned),
        np.abs(teeth_aligned - lips_aligned)
    ), np.abs(jaw_aligned - lips_aligned))
    
    alligator_sleeping = max_diff < (0.005 * close)  # lines intertwined
    alligator_awakening = max_diff > (0.015 * close)  # lines diverging
    
    # Trend direction: price above/below Alligator lines
    alligator_avg = (jaw_aligned + teeth_aligned + lips_aligned) / 3
    bullish_bias = close > alligator_avg
    bearish_bias = close < alligator_avg
    
    # Choppiness regime: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    chop_trending = chop_aligned < 38.2
    chop_ranging = chop_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic: Alligator awakening + volume spike + regime filter
        long_entry = False
        short_entry = False
        
        # Long: bullish bias + awakening + volume spike + (trending OR ranging)
        if bullish_bias and alligator_awakening[i]:
            long_entry = volume_spike_1d_aligned[i] and (chop_trending[i] or chop_ranging[i])
        
        # Short: bearish bias + awakening + volume spike + (trending OR ranging)
        if bearish_bias and alligator_awakening[i]:
            short_entry = volume_spike_1d_aligned[i] and (chop_trending[i] or chop_ranging[i])
        
        # Exit logic: Alligator sleeping again or opposite bias
        long_exit = alligator_sleeping[i] or (not bullish_bias and bearish_bias)
        short_exit = alligator_sleeping[i] or (bullish_bias and not bearish_bias)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_williams_alligator_vol_chop_v1"
timeframe = "12h"
leverage = 1.0