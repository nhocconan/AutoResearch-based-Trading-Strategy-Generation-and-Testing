#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + 1w chop regime filter.
# Long when price > Alligator teeth (smma8) and lips (smma5) > teeth (smma8) and volume > 2.0x 20-bar average.
# Short when price < Alligator teeth and lips < teeth and volume confirmation.
# Chop regime filter: only trade when 1w chop > 61.8 (range market) to avoid whipsaw in trends.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Target: 12-30 trades/year to minimize fee drag on 12h timeframe.

name = "12h_Williams_Alligator_Volume_Chop_1w_v1"
timeframe = "12h"
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
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator: SMMA(13,8), SMMA(8,5), SMMA(5,3) - jaws, teeth, lips
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close, 13)  # SMMA(13,8)
    teeth = smma(close, 8)  # SMMA(8,5)
    lips = smma(close, 5)   # SMMA(5,3)
    
    # Volume confirmation: current volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for volume spike calculation (HTF filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 12h
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data ONCE before loop for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Chop Index(14) for regime filter: >61.8 = range (mean revert), <38.2 = trending
    def true_range(h, l, c):
        return np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_chop_1w = true_range(high_1w, low_1w, close_1w)
    atr_14_1w = pd.Series(tr_chop_1w).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10(atr_14_1w / (highest_high_14_1w - lowest_low_14_1w)) / np.log10(14)
    
    # Align 1w chop to 12h
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 20  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(jaws[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average (12h) AND > 1.5x 1d average
        volume_confirm_12h = curr_volume > (vol_ma[i] * 2.0)
        volume_confirm_1d = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        volume_confirm = volume_confirm_12h and volume_confirm_1d
        
        # Alligator conditions
        lips_above_teeth = lips[i] > teeth[i]
        lips_below_teeth = lips[i] < teeth[i]
        price_above_teeth = curr_close > teeth[i]
        price_below_teeth = curr_close < teeth[i]
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1w_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: price > teeth AND lips > teeth AND volume confirmation AND chop regime
            if (price_above_teeth and 
                lips_above_teeth and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < teeth AND lips < teeth AND volume confirmation AND chop regime
            elif (price_below_teeth and 
                  lips_below_teeth and 
                  volume_confirm and 
                  chop_filter):
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
            # Exit: Alligator reverses (lips cross below teeth) OR chop regime ends (trending)
            elif lips[i] < teeth[i] or chop_1w_aligned[i] <= 61.8:
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
            # Exit: Alligator reverses (lips cross above teeth) OR chop regime ends (trending)
            elif lips[i] > teeth[i] or chop_1w_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals