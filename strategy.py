#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_ChopFilter_v1
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakout with 1d EMA50 trend filter and choppiness regime filter (CHOP > 50 = range) captures high-probability mean-reversion trades in ranging markets and breakout trades in trending markets. Uses discrete position sizing (0.25) and ATR-based stoploss (2.0) to target 50-150 total trades over 4 years. Works in both bull and bear markets by adapting to regime.
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
    
    # Get 1d data for trend filter (EMA50) and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need for EMA50 and CHOP
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR for choppiness calculation (TR and ATR)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d True Range for choppiness denominator (sum of TR over period)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    tr_sum_aligned = align_htf_to_ltf(prices, df_1d, tr_sum)
    
    # 1d High-Low range for choppiness numerator (max high - min low over period)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    range_hl_aligned = align_htf_to_ltf(prices, df_1d, range_hl)
    
    # Choppiness Index: CHOP = 100 * log10(sum(TRI)/log10(n)) / log10(range)
    # Simplified: CHOP = 100 * log10(tr_sum) / log10(range_hl) when range_hl > 0
    # We'll use: CHOP > 50 = ranging market (mean revert), CHOP < 50 = trending market (breakout)
    chop_raw = np.where(range_hl_aligned > 0, 100 * np.log10(tr_sum_aligned) / np.log10(range_hl_aligned), 50)
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)
    
    # Get 12h data for Camarilla calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for 12h
    # Camarilla levels based on previous bar's range
    # We need previous bar's OHLC for current bar's levels
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    # Set first value to NaN as there's no previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume filter: volume > 1.5x 20-period average (slightly looser for more trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA50 needs 50, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(chop_raw[i]) or 
            np.isnan(tr_sum_aligned[i]) or 
            np.isnan(range_hl_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        chop_val = chop_raw[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        
        # Get 12h close aligned for direct comparison
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        close_12h_val = close_12h_aligned[i]
        is_uptrend = close_12h_val > ema_50_val
        is_ranging = chop_val > 50.0  # CHOP > 50 = ranging market
        
        if position == 0:
            # Look for entry signals
            if is_ranging:
                # In ranging market: mean reversion at Camarilla extremes
                # Long when price touches S1 and starts reversing up
                # Short when price touches R1 and starts reversing down
                long_signal = (close_12h_val <= s1_val * 1.001) and (close_12h_val > low_12h[i]) and vol_spike[i]
                short_signal = (close_12h_val >= r1_val * 0.999) and (close_12h_val < high_12h[i]) and vol_spike[i]
            else:
                # In trending market: breakout in direction of trend
                # Long when price breaks above R1 in uptrend
                # Short when price breaks below S1 in downtrend
                long_signal = (close_12h_val > r1_val) and is_uptrend and vol_spike[i]
                short_signal = (close_12h_val < s1_val) and (not is_uptrend) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_12h_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_12h_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price touches opposite Camarilla level (S1 for longs) in ranging market
            # 2. Price closes below S1 (opposite Camarilla level) in trending market
            # 3. ATR-based stoploss: 2.0 * ATR below entry
            if is_ranging:
                exit_signal = close_12h_val >= s1_val * 0.999  # Touch S1 for mean reversion exit
            else:
                exit_signal = close_12h_val < s1_val  # Close below S1 for breakout exit
            stop_signal = close_12h_val < (entry_price - 2.0 * atr_val)
            if exit_signal or stop_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price touches opposite Camarilla level (R1 for shorts) in ranging market
            # 2. Price closes above R1 (opposite Camarilla level) in trending market
            # 3. ATR-based stoploss: 2.0 * ATR above entry
            if is_ranging:
                exit_signal = close_12h_val <= r1_val * 1.001  # Touch R1 for mean reversion exit
            else:
                exit_signal = close_12h_val > r1_val  # Close above R1 for breakout exit
            stop_signal = close_12h_val > (entry_price + 2.0 * atr_val)
            if exit_signal or stop_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0