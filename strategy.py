#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_v1
Hypothesis: Camarilla R1/S1 breakout on 4h with 1-day EMA34 trend filter and choppiness regime filter. Uses discrete position sizing (0.25) and ATR-based stoploss (2.5x) for risk management. Designed for low trade frequency (target 19-50/year) to minimize fee drag while capturing medium-term swings in both bull and bear markets. The 1-day EMA34 provides strong trend filtering that works across regimes, and choppiness filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d choppiness index (14-period)
    if len(df_1d) < 15:  # need at least 14 periods + 1 for TR
        chop_1d = np.full(len(df_1d), 50.0)  # neutral chop value
    else:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range for 1d
        tr1_1d = high_1d[1:] - low_1d[1:]
        tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
        tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Chop = 100 * log10(sumTR / (HH - LL)) / log10(14)
        range_1d = hh_1d - ll_1d
        chop_1d = np.where(
            (range_1d > 0) & ~np.isnan(tr_sum) & ~np.isnan(range_1d),
            100 * np.log10(tr_sum / range_1d) / np.log10(14),
            50.0  # neutral when range is zero or invalid
        )
        chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1d data for Camarilla calculation (previous bar)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift - use current bar as fallback for first bar
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(34), 1d chop calculation (14), ATR (14)
    start_idx = max(34, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
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
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        chop_filter = chop_1d_aligned[i] < 61.8  # trending regime (chop < 61.8)
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend up AND trending regime
            long_signal = (close_val > r1_aligned[i]) and trend_1d_up and chop_filter
            
            # Short: price breaks below S1 AND 1d trend down AND trending regime
            short_signal = (close_val < s1_aligned[i]) and trend_1d_down and chop_filter
            
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
            # Exit: trend flips down OR chop becomes too high (ranging) OR price hits ATR stoploss
            if (not trend_1d_up) or (chop_1d_aligned[i] >= 61.8) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR chop becomes too high (ranging) OR price hits ATR stoploss
            if (not trend_1d_down) or (chop_1d_aligned[i] >= 61.8) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0