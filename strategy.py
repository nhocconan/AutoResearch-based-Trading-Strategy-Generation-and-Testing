#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filtered_With_Volume_And_Chop_Regime
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with volume confirmation and Choppiness Index regime filter to avoid whipsaws.
Enter long when price > KAMA AND volume > 1.5x 20-day average AND CHOP > 61.8 (ranging).
Enter short when price < KAMA AND volume > 1.5x 20-day average AND CHOP > 61.8.
Exit on opposite signal. Uses 1d EMA34 as additional HTF trend filter from 1w timeframe.
Target: 30-100 trades over 4 years (7-25/year) with controlled risk and low fee drag.
Works in both bull and bear markets by adapting to volatility and avoiding false signals in strong trends.
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
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === 1d Indicators ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # KAMA(10, 2, 30) on 1d close
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # needs correction
    # Recalculate volatility properly: sum of absolute changes over ER period
    er_period = 10
    change_abs = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_sum = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close_1d[i-er_period:i+1])))
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change_abs / volatility_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume spike: volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_spike = vol_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Choppiness Index (CHOP) on 1d
    chop_period = 14
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first TR
    # Sum of TRUE RANGE over chop_period
    atr_sum = np.zeros_like(close_1d)
    for i in range(chop_period, len(close_1d)):
        atr_sum[i] = np.sum(tr_1d[i-chop_period+1:i+1])
    # Highest high and lowest low over chop_period
    hh_1d = np.zeros_like(close_1d)
    ll_1d = np.zeros_like(close_1d)
    for i in range(chop_period-1, len(close_1d)):
        hh_1d[i] = np.max(high_1d[i-chop_period+1:i+1])
        ll_1d[i] = np.min(low_1d[i-chop_period+1:i+1])
    # Avoid division by zero
    range_hl = hh_1d - ll_1d
    chop = np.where(range_hl > 0, 100 * np.log10(atr_sum / range_hl) / np.log10(chop_period), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 1w EMA34 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA calculation, volume MA, CHOP, EMA34
    start_idx = max(30, 20, 14, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        price_above_kama = close_val > kama_aligned[i]
        price_below_kama = close_val < kama_aligned[i]
        vol_spike = volume_spike_aligned[i]
        chop_high = chop_aligned[i] > 61.8  # ranging market
        trend_1w_up = close_val > ema_34_1w_aligned[i]
        trend_1w_down = close_val < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price > KAMA AND volume spike AND chop > 61.8 (ranging)
            # In ranging markets, mean reversion works; KAMA acts as dynamic mean
            long_signal = price_above_kama and vol_spike and chop_high
            
            # Short: price < KAMA AND volume spike AND chop > 61.8 (ranging)
            short_signal = price_below_kama and vol_spike and chop_high
            
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
            # Exit: price < KAMA OR chop drops below 38.2 (trending) OR 1w trend flips down
            if (price_below_kama) or (chop_aligned[i] < 38.2) or (not trend_1w_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > KAMA OR chop drops below 38.2 (trending) OR 1w trend flips up
            if (price_above_kama) or (chop_aligned[i] < 38.2) or (not trend_1w_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filtered_With_Volume_And_Chop_Regime"
timeframe = "1d"
leverage = 1.0