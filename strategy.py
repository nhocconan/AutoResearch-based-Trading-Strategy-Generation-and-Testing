#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Regime_v1
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and choppiness regime.
Long when price breaks above R1 AND 1d close > EMA(34) AND chop < 61.8 (trending).
Short when price breaks below S1 AND 1d close < EMA(34) AND chop < 61.8.
Volume confirmation required. Uses ATR-based stoploss via signal=0 on adverse moves.
Designed for fewer trades (target 50-150 over 4 years) to minimize fee drag and work in both bull/bear via trend filter.
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
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla, EMA, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C = (H+L+Close)/3 of previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    atr_14 = []
    tr_1d = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
    tr_1d = np.maximum(tr_1d, np.roll(df_1d['low'].values, 1))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    for i in range(len(df_1d)):
        if i < 14:
            atr_14.append(np.nan)
        else:
            atr_14.append(np.mean(tr_1d[i-13:i+1]))
    atr_14 = np.array(atr_14)
    
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(34), chop, volume MA(20), and need 1d data
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        chop_val = chop_1d_aligned[i]
        trending_regime = chop_val < 61.8  # trending market
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND 1d uptrend AND trending regime
            long_signal = (close_val > r1_aligned[i]) and vol_conf and trend_up and trending_regime
            
            # Short: price breaks below S1 AND volume confirm AND 1d downtrend AND trending regime
            short_signal = (close_val < s1_aligned[i]) and vol_conf and trend_down and trending_regime
            
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
            # Exit: ATR-based stoploss (2.0 * ATR) OR price drops below S1 (failed breakout) OR 1d trend flips down
            atr_20 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=20, min_periods=20).mean().values
            atr_val = atr_20[i] if not np.isnan(atr_20[i]) else 0.0
            stop_price = entry_price - 2.0 * atr_val
            if (close_val < stop_price) or (close_val < s1_aligned[i]) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR-based stoploss (2.0 * ATR) OR price rises above R1 (failed breakdown) OR 1d trend flips up
            atr_20 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=20, min_periods=20).mean().values
            atr_val = atr_20[i] if not np.isnan(atr_20[i]) else 0.0
            stop_price = entry_price + 2.0 * atr_val
            if (close_val > stop_price) or (close_val > r1_aligned[i]) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0