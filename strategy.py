#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_RegimeFilter
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1-week EMA50 trend filter and choppiness regime filter.
The 12h timeframe reduces trade frequency vs 4h to minimize fee drag while capturing significant moves.
Camarilla R1/S1 levels from prior 1-week provide strong intraday reversal points.
1-week EMA50 ensures we trade with the major trend, reducing whipsaws in ranging markets.
Choppiness filter (CHOP > 61.8) avoids trending markets where mean-reversion fails.
Discreet position sizing (0.25) limits drawdown. Target: 50-150 trades over 4 years.
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
    
    # Get 1w data for EMA trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.0 * 50-period average (stricter for 12h)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Camarilla levels from previous 1w bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1w['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1w['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1w['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate choppiness regime filter on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        chop_regime = np.zeros(n, dtype=bool)  # default to false if insufficient data
    else:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1_1d = high_1d[1:] - low_1d[1:]
        tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
        tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
        atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        
        # Chop = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
        sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        range_14 = max_high_14 - min_low_14
        
        # Avoid division by zero
        ratio = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
        chop = 100 * np.log10(ratio) / np.log10(14)
        chop_regime = chop > 61.8  # choppy market (mean revert)
        chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA(50), volume MA, ATR
    start_idx = max(50, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1w_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_1w_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        vol_spike = volume_spike[i]
        
        # Get chop regime (use aligned value)
        if len(df_1d) >= 20:
            chop_regime_val = chop_regime_aligned[i]
        else:
            chop_regime_val = False  # default to not choppy if insufficient data
        
        if position == 0:
            # Long: price breaks above R1 AND 1w trend up AND volume spike AND choppy regime
            long_signal = (close_val > r1_aligned[i]) and trend_1w_up and vol_spike and chop_regime_val
            
            # Short: price breaks below S1 AND 1w trend down AND volume spike AND choppy regime
            short_signal = (close_val < s1_aligned[i]) and trend_1w_down and vol_spike and chop_regime_val
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss OR chop regime ends
            if (not trend_1w_up) or (close_val < entry_price - 2.5 * atr[i]) or (not chop_regime_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss OR chop regime ends
            if (not trend_1w_down) or (close_val > entry_price + 2.5 * atr[i]) or (not chop_regime_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_RegimeFilter"
timeframe = "12h"
leverage = 1.0