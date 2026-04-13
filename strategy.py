#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and chop regime filter
    # Long: Close > H3 pivot AND volume > 1.5x 20-period avg AND chop < 61.8 (trending)
    # Short: Close < L3 pivot AND volume > 1.5x 20-period avg AND chop < 61.8 (trending)
    # Exit: Close crosses opposite pivot level (L3 for long, H3 for short) or chop > 61.8 (range)
    # Using 4h timeframe for optimal trade frequency, Camarilla for structure,
    # 12h volume for confirmation, chop for regime filter to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h volume moving average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        vol_ma_12h[i] = np.mean(vol_12h[i-20:i])
    volume_spike_12h = vol_12h > (1.5 * vol_ma_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's range)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # We'll use H3 and L3 as entry/exit levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng
    L3 = prev_close - 1.1 * rng
    
    # Align Camarilla levels to 4h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate chop regime filter on 4h data (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    
    atr_14 = np.full(n, np.nan)
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-14:i])
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i])
        lowest_low[i] = np.min(low[i-14:i])
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    sum_tr = np.full(n, np.nan)
    for i in range(14, n):
        sum_tr[i] = np.sum(tr[i-14:i])
    
    denominator = highest_high - lowest_low
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if denominator[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / denominator[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    chop_trending = chop < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or np.isnan(chop_trending[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation from 12h
        vol_confirm = volume_spike_12h_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor breakouts)
        regime_filter = chop_trending[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > H3_aligned[i]) and vol_confirm and regime_filter
        short_entry = (close[i] < L3_aligned[i]) and vol_confirm and regime_filter
        
        # Exit logic: opposite Camarilla level or chop > 61.8 (range)
        long_exit = (close[i] < L3_aligned[i]) or not regime_filter
        short_exit = (close[i] > H3_aligned[i]) or not regime_filter
        
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

name = "4h_12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0