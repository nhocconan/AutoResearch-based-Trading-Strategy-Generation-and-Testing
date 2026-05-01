#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + volume spike + 1d chop regime filter.
# Long when price breaks above Camarilla R3 with volume > 2x 20-bar average and chop > 61.8 (range).
# Short when price breaks below Camarilla S3 with volume confirmation and chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Session filter: 08-20 UTC to reduce noise. Target: 20-50 trades/year to minimize fee drag.
# Camarilla levels from 1d provide strong intraday support/resistance that work in both bull/bear via mean reversion in ranging markets.

name = "4h_Camarilla_R3S3_Volume_Chop_Regime_v1"
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
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar
    # R3 = close + 1.1*(high - low)/4
    # S3 = close - 1.1*(high - low)/4
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align 1d Camarilla levels to 4h (use previous day's levels for current 4h bar)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate Chop Index(14) for 1d regime filter: >61.8 = range (mean revert), <38.2 = trending
    def true_range(h, l, c):
        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - np.roll(c, 1)[1:])
        tr3 = np.abs(l[1:] - np.roll(c, 1)[1:])
        tr0 = np.max([h[0] - l[0], np.abs(h[0] - c[0]), np.abs(l[0] - c[0])])
        return np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    tr_chop_1d = true_range(high_1d, low_1d, close_1d)
    atr_14_1d = pd.Series(tr_chop_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_14_1d / (highest_high_14_1d - lowest_low_14_1d)) / np.log10(14)
    
    # Align 1d chop to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Camarilla breakout conditions
        breakout_up = curr_high > camarilla_r3_1d_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_1d_aligned[i]  # break below S3
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
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
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range (between S3 and R3) OR chop regime ends (trending)
            elif (curr_low <= camarilla_r3_1d_aligned[i] and curr_low >= camarilla_s3_1d_aligned[i]) or \
                 chop_1d_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range (between S3 and R3) OR chop regime ends (trending)
            elif (curr_high <= camarilla_r3_1d_aligned[i] and curr_high >= camarilla_s3_1d_aligned[i]) or \
                 chop_1d_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals