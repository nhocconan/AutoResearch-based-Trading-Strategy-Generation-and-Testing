#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d volume spike and chop regime filter.
# Long when KAMA(10,2,30) turns up with volume > 2.0x 20-bar average and 1d chop > 61.8 (range).
# Short when KAMA turns down with volume confirmation and chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# KAMA adapts to market noise, reducing whipsaw in choppy markets. Volume confirms conviction.
# Chop regime ensures mean-reversion edge in ranging markets (chop > 61.8).
# Works in bull (trend continuation with volume) and bear (mean reversion in chop) regimes.
# Target: 20-50 trades/year to avoid fee drag.

name = "4h_KAMA_10_2_30_Volume_1dChop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA(10,2,30) - ER period=10, fast=2, slow=30
    def calculate_kama(close, er_period=10, fast=2, slow=30):
        close_s = pd.Series(close)
        # Direction: absolute net change over er_period
        direction = np.abs(close_s.diff(er_period))
        # Volatility: sum of absolute daily changes over er_period
        volatility = close_s.diff().abs().rolling(window=er_period, min_periods=1).sum()
        # Efficiency Ratio
        er = np.where(volatility != 0, direction / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_period] = close_s.iloc[er_period]  # seed with close
        # Calculate KAMA
        for i in range(er_period + 1, len(close)):
            if not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = np.nan
    kama_up = kama > kama_prev
    kama_down = kama < kama_prev
    
    # Load 1d data ONCE before loop for chop regime and volume spike (HTF filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Chop Index(14) for 1d: >61.8 = range (mean revert), <38.2 = trending
    def true_range(h, l, c):
        h_l = h - l
        h_pc = np.abs(np.subtract(h, np.roll(c, 1)))
        l_pc = np.abs(np.subtract(l, np.roll(c, 1)))
        h_pc[0] = np.abs(h[0] - c[0])
        l_pc[0] = np.abs(l[0] - c[0])
        return np.maximum(h_l, np.maximum(h_pc, l_pc))
    
    tr_chop_1d = true_range(high_1d, low_1d, close_1d)
    atr_14_1d = pd.Series(tr_chop_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denom_1d = highest_high_14_1d - lowest_low_14_1d
    chop_1d = np.where(denom_1d != 0, 100 * np.log10(atr_14_1d / denom_1d) / np.log10(14), 50.0)
    
    # Calculate 1d volume spike: current volume > 2.0x 20-bar average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = np.where(vol_ma_1d > 0, volume_1d > (vol_ma_1d * 2.0), False)
    
    # Align 1d indicators to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for KAMA
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(kama_prev[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-bar average (4h)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0 or np.isnan(vol_ma):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.8)
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA turning up AND volume confirmation AND chop regime AND 1d volume spike
            if (kama_up[i] and 
                volume_confirm and 
                chop_filter and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: KAMA turning down AND volume confirmation AND chop regime AND 1d volume spike
            elif (kama_down[i] and 
                  volume_confirm and 
                  chop_filter and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns down OR chop regime ends (trending)
            elif (kama_down[i] or chop_1d_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns up OR chop regime ends (trending)
            elif (kama_up[i] or chop_1d_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals