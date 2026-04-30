#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1d HMA21 trend filter and volume confirmation.
# Long when price breaks above H4 with uptrend (price > 1d HMA21) and volume spike.
# Short when price breaks below L4 with downtrend (price < 1d HMA21) and volume spike.
# Uses ATR trailing stop (1.5x) for risk management.
# Targets 75-200 trades over 4 years (19-50/year) with discrete position sizing (0.25).
# H4/L4 levels provide stronger breakout signals than R3/S3, reducing false entries.
# Works in both bull/bear markets by requiring 1d HMA21 trend alignment.

name = "4h_Camarilla_H4L4_Breakout_1dHMA21_Trend_VolumeSpike_ATRTrail_v1"
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
    
    # Load 1d data ONCE before loop for HMA21 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA21 for trend filter
    close_1d = df_1d['close'].values
    half_n = int(21 / 2)
    sqrt_n = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA calculation
    wma_half = np.array([wma(close_1d[i-half_n+1:i+1], half_n) if i >= half_n-1 else np.nan for i in range(len(close_1d))])
    wma_full = np.array([wma(close_1d[i-21+1:i+1], 21) if i >= 20 else np.nan for i in range(len(close_1d))])
    hma_raw = 2 * wma_half - wma_full
    hma_21 = np.array([wma(hma_raw[i-sqrt_n+1:i+1], sqrt_n) if i >= sqrt_n-1 else np.nan for i in range(len(hma_raw))])
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: H4 = close + ((high-low)*1.1/2), L4 = close - ((high-low)*1.1/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    rng = high_1d - low_1d
    camarilla_h4 = close_1d_arr + (rng * 1.1 / 2)
    camarilla_l4 = close_1d_arr - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 34  # warmup for HMA21 and ATR
    
    for i in range(start_idx, n):
        # Skip if Camarilla levels not available
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 1d HMA21 determines trend direction
        is_uptrend = close[i] > hma_21_aligned[i]
        is_downtrend = close[i] < hma_21_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend and curr_close > h4_aligned[i] and curr_volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif is_downtrend and curr_close < l4_aligned[i] and curr_volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 1.5 * ATR below highest since entry
            if curr_close < highest_since_entry - 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 1.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals