#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d volume spike + 1w chop regime filter.
# Long when price breaks above Camarilla R3 with volume > 2.0x 20-bar average and weekly chop > 61.8 (range).
# Short when price breaks below Camarilla S3 with volume confirmation and weekly chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Session filter: 08-20 UTC to reduce noise. Target: 12-37 trades/year to minimize fee drag.
# This strategy targets range-bound markets where Camarilla levels act as support/resistance,
# with volume confirmation to avoid false breakouts and weekly chop filter to ensure ranging conditions.

name = "12h_Camarilla_R3S3_Breakout_1dVolume_1wChop_v1"
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
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (R3, S3) from previous 12h bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # We use the previous bar's high/low/close to avoid look-ahead
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Load 1d data ONCE before loop for volume confirmation (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-bar average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 12h
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data ONCE before loop for chop regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Chop Index(14) for regime filter: >61.8 = range (mean revert), <38.2 = trending
    def true_range(h, l, c):
        # Handle first element where roll gives last element (we'll fix after)
        h_l = h - l
        h_pc = np.abs(h - np.roll(c, 1))
        l_pc = np.abs(l - np.roll(c, 1))
        tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
        # Fix first element: use current bar only
        tr[0] = h[0] - l[0]
        return tr
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_chop_1w = true_range(high_1w, low_1w, close_1w)
    atr_14_1w = pd.Series(tr_chop_1w).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_14_1w = highest_high_14_1w - lowest_low_14_1w
    chop_1w = np.zeros_like(atr_14_1w)
    mask = range_14_1w > 0
    chop_1w[mask] = 100 * np.log10(atr_14_1w[mask] / range_14_1w[mask]) / np.log10(14)
    
    # Align 1w chop to 12h
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 20  # warmup for volume MA and Camarilla
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1w_aligned[i]) or
            vol_ma_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 1d volume average
        volume_confirm = curr_volume > (vol_ma_1d_aligned[i] * 2.0)
        
        # Camarilla breakout conditions
        breakout_up = curr_high > camarilla_r3[i]  # break above R3
        breakout_down = curr_low < camarilla_s3[i]  # break below S3
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1w_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up AND volume confirmation AND chop regime
            if (breakout_up and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla breakout down AND volume confirmation AND chop regime
            elif (breakout_down and 
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
            # Exit: price re-enters Camarilla range (between S3 and R3) OR chop regime ends (trending)
            elif (curr_low >= camarilla_s3[i] and curr_low <= camarilla_r3[i]) or \
                 chop_1w_aligned[i] <= 61.8:
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
            # Exit: price re-enters Camarilla range (between S3 and R3) OR chop regime ends (trending)
            elif (curr_high >= camarilla_s3[i] and curr_high <= camarilla_r3[i]) or \
                 chop_1w_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals