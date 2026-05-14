#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike (>2x 20-bar average), and choppiness regime filter (CHOP > 50 = range, avoid strong trends).
Combines price channel breakout with trend confirmation, volume confirmation, and regime filter to reduce false breakouts.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
Designed to work in both bull and bear markets by using daily trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    camarilla_h4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_l4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume (strict filter to reduce trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Choppiness regime filter: avoid strong trends (CHOP > 50 = range/choppy market)
    atr_period = 14
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    price_range = highest_high - lowest_low
    chop_raw = np.zeros_like(close)
    mask = price_range > 0
    atr_sum = pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop_raw[mask] = 100 * np.log10(atr_sum[mask]) / np.log10(atr_period) / np.log10(price_range[mask])
    chop_raw = np.where(np.isnan(chop_raw), 50, chop_raw)
    
    chop_filter = chop_raw > 50  # Only trade in choppy/range markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 and ATR
    start_idx = max(35, atr_period * 2)  # EMA34 needs 34, ATR needs 28 for sum
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop_raw[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1d trend with volume confirmation and chop filter
            # Long: price breaks above R1 in uptrend (close > EMA34) with volume spike and chop filter
            # Short: price breaks below S1 in downtrend (close < EMA34) with volume spike and chop filter
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema34_aligned[i]) and volume_spike[i] and chop_filter[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema34_aligned[i]) and volume_spike[i] and chop_filter[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop"
timeframe = "4h"
leverage = 1.0