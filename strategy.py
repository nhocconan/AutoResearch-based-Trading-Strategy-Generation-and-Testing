#!/usr/bin/env python3
# 12h_KAMA_RSI_Chop_Filter_v1
# Hypothesis: KAMA identifies adaptive trend direction on 12h, RSI(14) provides momentum filter, and Choppiness Index (14) filters ranging markets.
# In trending markets (CHOP < 38.2): go long when KAMA up & RSI > 50, short when KAMA down & RSI < 50.
# In ranging markets (CHOP > 61.8): mean revert at Bollinger Bands (20,2) - long at lower band, short at upper band.
# Uses 1d trend filter (EMA34) to avoid counter-trend trades. Designed for low turnover (<30 trades/year) to minimize fee drag.

name = "12h_KAMA_RSI_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # KAMA on 12h close
    def calculate_kama(close_arr, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close_arr, n=period))
        volatility = np.sum(np.abs(np.diff(close_arr)), axis=0)
        er = np.zeros_like(close_arr)
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_arr)
        kama[:period] = close_arr[:period]
        for i in range(period, len(close_arr)):
            kama[i] = kama[i-1] + sc[i] * (close_arr[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    kama_dir = np.zeros_like(close)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, -1)
    
    # RSI(14)
    def calculate_rsi(close_arr, period=14):
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_arr)
        avg_loss = np.zeros_like(close_arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close_arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Choppiness Index (14)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr = max(high_arr[i] - low_arr[i], 
                     abs(high_arr[i] - close_arr[i-1]),
                     abs(low_arr[i] - close_arr[i-1]))
            atr[i] = tr
        # smoothed ATR
        atr_sum = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # highest high and lowest low over period
        highest_high = np.zeros_like(close_arr)
        lowest_low = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            highest_high[i] = np.max(high_arr[i-period+1:i+1])
            lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        chop = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Bollinger Bands (20,2) for mean reversion in ranging markets
    bb_period = 20
    bb_std = 2
    sma = np.zeros_like(close)
    for i in range(bb_period-1, len(close)):
        sma[i] = np.mean(close[i-bb_period+1:i+1])
    bb_std_dev = np.zeros_like(close)
    for i in range(bb_period-1, len(close)):
        bb_std_dev[i] = np.std(close[i-bb_period+1:i+1])
    bb_upper = sma + bb_std_dev * bb_std
    bb_lower = sma - bb_std_dev * bb_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(34, 30, 14, 20)  # EMA34, KAMA, RSI, CHOP, BB
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Market regime
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # Long entry conditions
            long_signal = False
            if is_trending and uptrend_1d:
                # Trending up: KAMA up and RSI > 50
                if kama_dir[i] == 1 and rsi[i] > 50:
                    long_signal = True
            elif is_ranging:
                # Ranging: mean reversion at lower Bollinger Band
                if close[i] <= bb_lower[i]:
                    long_signal = True
            
            # Short entry conditions
            short_signal = False
            if is_trending and downtrend_1d:
                # Trending down: KAMA down and RSI < 50
                if kama_dir[i] == -1 and rsi[i] < 50:
                    short_signal = True
            elif is_ranging:
                # Ranging: mean reversion at upper Bollinger Band
                if close[i] >= bb_upper[i]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            if is_trending:
                # Exit trend: KAMA down or RSI < 40
                if kama_dir[i] == -1 or rsi[i] < 40:
                    exit_signal = True
            else:  # ranging or choppy
                # Exit mean reversion: price crosses SMA
                if close[i] >= sma[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            if is_trending:
                # Exit trend: KAMA up or RSI > 60
                if kama_dir[i] == 1 or rsi[i] > 60:
                    exit_signal = True
            else:  # ranging or choppy
                # Exit mean reversion: price crosses SMA
                if close[i] <= sma[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals