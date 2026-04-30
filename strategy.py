#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
# In trending markets (price > 1d EMA50), break above/below 12h Donchian(20) with volume triggers continuation entries.
# Uses ATR-based trailing stop (2.0x) to manage risk. Targets 20-40 trades/year to minimize fee drag.
# Works in bull/bear via trend filter: only trades in direction of 1d EMA50.

name = "4h_12hDonchian20_1dEMA50_Trend_VolumeSpike_ATRTrail_v1"
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
    
    # Load 12h data ONCE before loop for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Rolling max/min for Donchian upper/lower bands
    high_ma_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_ma_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_ma_20)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h ATR(14) for dynamic trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Regime filter: price above/below 1d EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long breakouts above Donchian high with volume
                if curr_close > curr_donchian_high and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
            elif is_downtrend:
                # In downtrend: look for short breakdowns below Donchian low with volume
                if curr_close < curr_donchian_low and curr_volume_spike:
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