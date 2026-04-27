#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Choppiness Index regime filter + Donchian breakout with volume confirmation.
# Long when: price breaks above Donchian(20) upper, market is trending (CHOP < 38.2), volume > 1.5x avg, and 1-day close > 1-week EMA20 (bull bias).
# Short when: price breaks below Donchian(20) lower, market is trending (CHOP < 38.2), volume > 1.5x avg, and 1-day close < 1-week EMA20 (bear bias).
# Exit when price crosses Donchian middle or CHOP > 61.8 (range regime).
# Target: 12-37 trades/year to avoid fee drag. Works in bull/bear via regime-adaptive trend following.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for longer-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channel (20) on 12h data
    dc_period = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(dc_period - 1, n):
        dc_upper[i] = np.max(high[i - dc_period + 1:i + 1])
        dc_lower[i] = np.min(low[i - dc_period + 1:i + 1])
    
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Calculate Choppiness Index (14) on 1d data
    chop_period = 14
    atr_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= chop_period:
        # Calculate True Range for 1d
        tr = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            hl = df_1d['high'].values[i] - df_1d['low'].values[i]
            hc = abs(df_1d['high'].values[i] - close_1d[i-1])
            lc = abs(df_1d['low'].values[i] - close_1d[i-1])
            tr[i] = max(hl, hc, lc)
        tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
        
        # Calculate ATR
        atr_1d[chop_period - 1] = np.mean(tr[1:chop_period])
        for i in range(chop_period, len(tr)):
            atr_1d[i] = (tr[i] + (chop_period - 1) * atr_1d[i-1]) / chop_period
        
        # Calculate Chop
        sum_atr = np.full(len(close_1d), np.nan)
        for i in range(chop_period - 1, len(close_1d)):
            sum_atr[i] = np.sum(atr_1d[i - chop_period + 1:i + 1])
        
        max_high = np.full(len(close_1d), np.nan)
        min_low = np.full(len(close_1d), np.nan)
        for i in range(chop_period - 1, len(close_1d)):
            max_high[i] = np.max(df_1d['high'].values[i - chop_period + 1:i + 1])
            min_low[i] = np.min(df_1d['low'].values[i - chop_period + 1:i + 1])
        
        chop_1d = np.full(len(close_1d), 50.0)  # default neutral
        for i in range(chop_period - 1, len(close_1d)):
            if max_high[i] > min_low[i] and sum_atr[i] > 0:
                chop_1d[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(chop_period)
    
    # Calculate 1-week EMA20 for trend filter
    ema_1w_period = 20
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_1w_period:
        ema_1w[ema_1w_period - 1] = np.mean(close_1w[:ema_1w_period])
        for i in range(ema_1w_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_1w_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_1w_period + 1))))
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1d indicators to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Align 1w EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20), Chop(14), EMA20(1w), and volume MA20
    start_idx = max(dc_period - 1, chop_period - 1, ema_1w_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Regime filters
        trending_market = chop_1d_aligned[i] < 38.2  # CHOP < 38.2 = trending
        ranging_market = chop_1d_aligned[i] > 61.8   # CHOP > 61.8 = ranging
        
        # Trend bias from 1-week EMA
        bull_bias = close_1d[-1] > ema_1w_aligned[i] if i < len(ema_1w_aligned) else False
        bear_bias = close_1d[-1] < ema_1w_aligned[i] if i < len(ema_1w_aligned) else False
        
        if position == 0:
            # Long: break above Donchian upper + trending + volume + bull bias
            if (price > dc_upper[i] and trending_market and vol_filter and bull_bias):
                signals[i] = size
                position = 1
            # Short: break below Donchian lower + trending + volume + bear bias
            elif (price < dc_lower[i] and trending_market and vol_filter and bear_bias):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR market becomes ranging
            if price < dc_middle[i] or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR market becomes ranging
            if price > dc_middle[i] or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_ChopRegime_DonchianBreakout_Volume_1wEMA"
timeframe = "12h"
leverage = 1.0