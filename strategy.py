#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Uses 1w Camarilla pivot levels (R4/S4) for regime: above weekly R4 = bull, below weekly S4 = bear, between = range.
# In bull regime: long on Donchian(20) breakout with volume confirmation.
# In bear regime: short on Donchian(20) breakdown with volume confirmation.
# In range regime: fade at weekly R3/S3 levels with volume confirmation.
# Uses ATR-based trailing stop (2.0x) for risk management.
# Designed for low trade frequency (~12-37/year) to minimize fee drag on 6h timeframe.

name = "6h_1wCamarilla_Donchian20_VolumeSpike_ATRTrail_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w) / 2
    camarilla_r4 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_s4 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align 1w Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Calculate 6h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h ATR(14) for dynamic trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Regime filter based on weekly Camarilla levels
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_upper = highest_20[i]
        curr_lower = lowest_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Bull regime: price above weekly R4
            if curr_close > curr_r4:
                # Look for long on Donchian breakout with volume
                if curr_close > curr_upper and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
            
            # Bear regime: price below weekly S4
            elif curr_close < curr_s4:
                # Look for short on Donchian breakdown with volume
                if curr_close < curr_lower and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
            
            # Range regime: price between weekly S4 and R4
            else:
                # Look for mean reversion at weekly R3/S3 with volume
                if curr_close < curr_s3 and curr_volume_spike:
                    # Oversold: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                elif curr_close > curr_r3 and curr_volume_spike:
                    # Overbought: short
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals