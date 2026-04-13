#!/usr/bin/env python3
"""
12h_1d1w_Vortex_Trend_Filter
Hypothesis: 12h price action with 1d Vortex trend filter and 1w volatility regime.
Long when price > 12h EMA21 + VI+ > VI- (bullish vortex) + 1w ATR% < 0.08 (low vol regime).
Short when price < 12h EMA21 + VI- > VI+ (bearish vortex) + 1w ATR% < 0.08 (low vol regime).
Exit when vortex reverses or price crosses EMA21.
Designed for 12h to capture trends in low-volatility regimes, reducing whipsaw in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 12h EMA21
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).values
    
    # 1d Vortex Indicator (VI+ and VI-)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Vortex movements
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # 1w ATR% (volatility regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1_w = np.abs(high_1w[1:] - low_1w[1:])
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w = np.concatenate([[np.nan], tr_w])
    
    atr14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_percent_w = atr14_w / close_1w
    atr_percent_w_aligned = align_htf_to_ltf(prices, df_1w, atr_percent_w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_21[i]) or np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or
            np.isnan(atr_percent_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime (1w ATR% < 8%)
        low_vol = atr_percent_w_aligned[i] < 0.08
        
        # Vortex conditions
        bullish_vortex = vi_plus_aligned[i] > vi_minus_aligned[i]
        bearish_vortex = vi_minus_aligned[i] > vi_plus_aligned[i]
        
        # EMA21 conditions
        price_above_ema = close[i] > ema_21[i]
        price_below_ema = close[i] < ema_21[i]
        
        # Entry conditions
        long_entry = price_above_ema and bullish_vortex and low_vol
        short_entry = price_below_ema and bearish_vortex and low_vol
        
        # Exit conditions
        long_exit = (price_below_ema or not bullish_vortex or not low_vol)
        short_exit = (price_above_ema or not bearish_vortex or not low_vol)
        
        if position == 0:
            if long_entry:
                position = 1
                signals[i] = position_size
            elif short_entry:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d1w_Vortex_Trend_Filter"
timeframe = "12h"
leverage = 1.0