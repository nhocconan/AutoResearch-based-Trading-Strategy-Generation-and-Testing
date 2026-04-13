#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Long when: price breaks above Camarilla H3 (1d) AND 1d volume > 1.5x 20-day avg AND chop > 61.8 (trending)
    # Short when: price breaks below Camarilla L3 (1d) AND 1d volume > 1.5x 20-day avg AND chop > 61.8 (trending)
    # Exit when: price returns to Camarilla Pivot level (mean reversion) OR chop < 38.2 (range)
    # Uses discrete sizing (0.25) targeting 75-200 trades over 4 years.
    # Works in bull/bear via volume spike confirmation and chop regime filter preventing false breakouts.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla levels, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for 1d (based on previous day)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Pivot = (high + low + close)/3
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses same bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_h3 = camarilla_pivot + 1.1 * camarilla_range * 1.1 / 4
    camarilla_l3 = camarilla_pivot - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1d volume spike (current volume > 1.5x 20-day average)
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * volume_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 1d Chopiness Index (trending regime filter)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    tr14 = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    tr14[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr14).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = max_high - min_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (np.log10(14) * denominator))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        breakout_long = close[i] > camarilla_h3_aligned[i]
        breakout_short = close[i] < camarilla_l3_aligned[i]
        vol_confirmed = volume_spike_aligned[i] > 0.5  # bool aligned as float
        trending_regime = chop_aligned[i] > 61.8
        mean_reversion = (abs(close[i] - camarilla_pivot_aligned[i]) < 0.001 * camarilla_pivot_aligned[i])  # near pivot
        ranging_regime = chop_aligned[i] < 38.2
        
        # Entry conditions
        long_entry = breakout_long and vol_confirmed and trending_regime and position != 1
        short_entry = breakout_short and vol_confirmed and trending_regime and position != -1
        
        # Exit conditions
        exit_long = mean_reversion or ranging_regime or (position == 1 and not trending_regime)
        exit_short = mean_reversion or ranging_regime or (position == -1 and not trending_regime)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0