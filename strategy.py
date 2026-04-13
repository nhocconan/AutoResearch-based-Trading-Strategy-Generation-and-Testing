#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above H3 (1d) AND volume > 1.5x 20-period average AND chop < 61.8 (trending)
    # Short: price breaks below L3 (1d) AND volume > 1.5x 20-period average AND chop < 61.8 (trending)
    # Exit: price returns to pivot point (PP) or chop > 61.8 (choppy regime)
    # Using 12h timeframe for low trade frequency, Camarilla from 1d for structure,
    # volume for confirmation, chop regime to avoid whipsaws.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 2
    # L3 = PP - (H - L) * 1.1 / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r = high_1d - low_1d
    h3 = pp + (r * 1.1 / 2.0)
    l3 = pp - (r * 1.1 / 2.0)
    
    # Align daily Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 12h chop regime (Ehler's Chop Index)
    # Chop = 100 * log10(sum(ATR(14)) / log10((max(high,n) - min(low,n)) * sqrt(n)))
    # We'll use a simplified version: Chop = 100 * (true_range_sum / (max_high - min_low) * sqrt(period))
    period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # True range sum over period
    tr_sum = np.full(n, np.nan)
    for i in range(period, n):
        tr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Max high and min low over period
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(period-1, n):
        max_high[i] = np.max(high[i-period+1:i+1])
        min_low[i] = np.min(low[i-period+1:i+1])
    
    # Chop index
    chop = np.full(n, np.nan)
    for i in range(period, n):
        if max_high[i] > min_low[i] and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (np.log10((max_high[i] - min_low[i]) * np.sqrt(period))))
        else:
            chop[i] = 50.0  # neutral
    
    # Volume confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 61.8 = trending (favor breakouts)
        trending_regime = chop[i] < 61.8
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > h3_aligned[i]) and vol_confirm and trending_regime
        short_entry = (close[i] < l3_aligned[i]) and vol_confirm and trending_regime
        
        # Exit logic: return to pivot or choppy regime
        long_exit = (close[i] < pp_aligned[i]) or (chop[i] >= 61.8)
        short_exit = (close[i] > pp_aligned[i]) or (chop[i] >= 61.8)
        
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

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0