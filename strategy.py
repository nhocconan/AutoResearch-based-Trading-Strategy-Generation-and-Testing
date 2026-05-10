#!/usr/bin/env python3
"""
4h_Engulfing_1dTrend_Volume
Hypothesis: On 4h timeframe, bullish/bearish engulfing candles aligned with 1d EMA50 trend and volume spike provide high-probability entries. 
Engulfing patterns signal strong momentum shifts, and when combined with daily trend and volume confirmation, they work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Uses 1d EMA50 for trend filter and 1d volume spike for confirmation. Targets 20-40 trades per year to minimize fee drag.
"""

name = "4h_Engulfing_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume average for spike detection
    volume_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_avg_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_avg_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need EMA50 warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 2x average 1d volume (scaled)
        # 6 four-hour periods in 1 day, so divide daily average by 6
        vol_4h_avg = vol_avg_1d_aligned[i] / 6.0
        volume_spike = volume[i] > 2.0 * vol_4h_avg
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and \
                         (open_price[i-1] > close[i-1]) and \
                         (close[i] >= open_price[i-1]) and \
                         (open_price[i] <= close[i-1])
        
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and \
                         (open_price[i-1] < close[i-1]) and \
                         (open_price[i] >= close[i-1]) and \
                         (close[i] <= open_price[i-1])
        
        if position == 0:
            # Long: bullish engulfing in uptrend with volume spike
            if bullish_engulf and close[i] > ema50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing in downtrend with volume spike
            elif bearish_engulf and close[i] < ema50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend reversal or opposite engulfing
            if close[i] < ema50_1d_aligned[i] or bearish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend reversal or opposite engulfing
            if close[i] > ema50_1d_aligned[i] or bullish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals