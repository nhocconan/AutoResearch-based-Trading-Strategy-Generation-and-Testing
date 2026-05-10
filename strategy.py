#!/usr/bin/env python3
# 6H_OrderBlock_Breaker_BullBear
# Hypothesis: Identifies institutional order blocks (OB) on 1d chart, then enters long when price breaks above bullish OB in bullish market (price > weekly EMA50), short when breaks below bearish OB in bearish market (price < weekly EMA50). Uses volume > 1.5x 20-period average for confirmation. Designed for low trade frequency (~15-30/year) with discrete sizing (0.25) to minimize fee scrub. Works in bull/bear by aligning with weekly trend.

name = "6H_OrderBlock_Breaker_BullBear"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for order blocks
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Identify bullish and bearish order blocks on 1d
    # Bullish OB: last down candle before strong up move (close < open, then next candle closes above its high)
    # Bearish OB: last up candle before strong down move (close > open, then next candle closes below its low)
    bullish_ob_low = np.full(len(df_1d), np.nan)
    bullish_ob_high = np.full(len(df_1d), np.nan)
    bearish_ob_low = np.full(len(df_1d), np.nan)
    bearish_ob_high = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)-1):
        # Bullish OB: candle i is down, candle i+1 is up and closes above candle i's high
        if df_1d['close'].iloc[i] < df_1d['open'].iloc[i] and \
           df_1d['close'].iloc[i+1] > df_1d['open'].iloc[i+1] and \
           df_1d['close'].iloc[i+1] > df_1d['high'].iloc[i]:
            bullish_ob_low[i] = df_1d['low'].iloc[i]
            bullish_ob_high[i] = df_1d['high'].iloc[i]
        # Bearish OB: candle i is up, candle i+1 is down and closes below candle i's low
        elif df_1d['close'].iloc[i] > df_1d['open'].iloc[i] and \
             df_1d['close'].iloc[i+1] < df_1d['open'].iloc[i+1] and \
             df_1d['close'].iloc[i+1] < df_1d['low'].iloc[i]:
            bearish_ob_low[i] = df_1d['low'].iloc[i]
            bearish_ob_high[i] = df_1d['high'].iloc[i]
    
    # Align order blocks to 6h timeframe
    bullish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_low)
    bullish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_high)
    bearish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_low)
    bearish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_high)
    
    # Weekly trend filter: EMA 50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(bullish_ob_low_aligned[i]) or np.isnan(bearish_ob_low_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above bullish OB high + volume confirmation + weekly uptrend
            if (close[i] > bullish_ob_high_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below bearish OB low + volume confirmation + weekly downtrend
            elif (close[i] < bearish_ob_low_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below bullish OB low (invalidation)
            if close[i] < bullish_ob_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above bearish OB high (invalidation)
            if close[i] > bearish_ob_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals