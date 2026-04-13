#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
    # Long: price breaks above H3 (bullish bias) AND volume > 1.5x 20-period avg AND CHOP > 61.8 (range)
    # Short: price breaks below L3 (bearish bias) AND volume > 1.5x 20-period avg AND CHOP > 61.8 (range)
    # Exit: price returns to PIVOT level or volume dry-up
    # Using 4h timeframe for optimal trade frequency, Camarilla for structure,
    # 1d volume for confirmation, CHOP for regime filter (mean reversion in chop).
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # PIVOT = (high + low + close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align to 4h
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Daily volume spike (>1.5x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Choppiness Index (CHOP) on 4h - range regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    atr_period = 14
    chop_period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # CHOP calculation
    chop = np.full(n, np.nan)
    for i in range(chop_period, n):
        atr_sum = np.sum(tr[i-chop_period+1:i+1])
        max_high = np.max(high[i-chop_period+1:i+1])
        min_low = np.min(low[i-chop_period+1:i+1])
        range_val = max_high - min_low
        if range_val > 0:
            chop[i] = 100 * np.log10(atr_sum) / (np.log10(chop_period) * np.log10(range_val))
        else:
            chop[i] = 50  # neutral
    
    # Regime filter: CHOP > 61.8 = ranging (mean reversion favorable)
    chop_regime = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(pivot_4h[i]) or 
            np.isnan(volume_spike_4h[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike_4h[i] > 0.5  # boolean array converted to float
        
        # Regime filter: only trade in ranging markets
        regime_ok = chop_regime[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > h3_4h[i]) and vol_confirm and regime_ok
        short_entry = (close[i] < l3_4h[i]) and vol_confirm and regime_ok
        
        # Exit logic: return to pivot or volume dry-up
        long_exit = (close[i] < pivot_4h[i]) or not vol_confirm
        short_exit = (close[i] > pivot_4h[i]) or not vol_confirm
        
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

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0