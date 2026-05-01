#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA50 trend + volume confirmation + chop regime filter.
# Long when price breaks above Camarilla R3 AND price > 1d EMA50 AND volume > 1.8x 12h volume median AND chop > 61.8 (range).
# Short when price breaks below Camarilla S3 AND price < 1d EMA50 AND volume > 1.8x 12h volume median AND chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to minimize fee drag.
# Camarilla R3/S3 provide reliable breakout levels. 1d EMA50 offers smooth trend filter.
# Volume spike confirms institutional interest. Chop > 61.8 ensures mean-reversion logic works in ranging markets.
# Works in both bull/bear: breakouts capture trends, chop filter avoids false signals in strong trends.

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Volume_Chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume median (30-period for stability)
    vol_median_12h = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Calculate Choppiness Index (14) for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        atr_sum = pd.Series(np.maximum(high_arr - low_arr, 
                                       np.maximum(np.abs(high_arr - np.roll(close_arr, 1)), 
                                                np.abs(low_arr - np.roll(close_arr, 1))))).rolling(window).sum()
        max_high = pd.Series(high_arr).rolling(window).max()
        min_low = pd.Series(low_arr).rolling(window).min()
        chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(window)
        # Handle first value
        tr_first = np.max([high_arr[0] - low_arr[0], 
                          np.abs(high_arr[0] - close_arr[0]), 
                          np.abs(low_arr[0] - close_arr[0])])
        atr_sum_first = tr_first
        chop_values = chop.values
        chop_values[0] = 50.0  # neutral
        return chop_values
    
    chop = calculate_chop(high, low, close, 14)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_median_12h[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 12h volume median
        if vol_median_12h[i] <= 0 or np.isnan(vol_median_12h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_12h[i] * 1.8)
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion at pivots)
        ranging = chop[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla R3 AND uptrend AND volume confirmation AND ranging
            if (curr_high > camarilla_r3_aligned[i] and 
                uptrend and 
                volume_confirm and 
                ranging):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Camarilla S3 AND downtrend AND volume confirmation AND ranging
            elif (curr_low < camarilla_s3_aligned[i] and 
                  downtrend and 
                  volume_confirm and 
                  ranging):
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
            # Exit: price breaks below Camarilla S3 OR trend turns down OR chop < 38.2 (strong trend)
            elif (curr_low < camarilla_s3_aligned[i]) or (not uptrend) or (chop[i] < 38.2):
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
            # Exit: price breaks above Camarilla R3 OR trend turns up OR chop < 38.2 (strong trend)
            elif (curr_high > camarilla_r3_aligned[i]) or (not downtrend) or (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals