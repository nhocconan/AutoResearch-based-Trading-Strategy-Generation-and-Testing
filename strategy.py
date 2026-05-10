#!/usr/bin/env python3
"""
12h_Engulfing_Candle_1dTrend_Volume
Hypothesis: Engulfing candlestick patterns combined with 1d EMA21 trend filter and volume confirmation.
Engulfing candles signal potential reversals; in trending markets they often mark pullback ends.
Works in bull markets (bullish engulfing after pullbacks in uptrend) and bear markets (bearish engulfing after bounces in downtrend).
Volume confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_Engulfing_Candle_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema21_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 21:
        ema21_1d[20] = np.mean(close_1d[:21])
        alpha = 2 / (21 + 1)
        for i in range(21, len(close_1d)):
            ema21_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema21_1d[i-1]
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # warmup for EMA calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema21_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume from 1d
        # Approximate 12h volume from 1d: 1d volume / 2 (since 24h/12h = 2)
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        # Engulfing candle detection
        bullish_engulf = (close[i] > open_[i]) and (open_[i-1] > close[i-1]) and (close[i] > open_[i-1]) and (open_[i] < close[i-1])
        bearish_engulf = (close[i] < open_[i]) and (open_[i-1] < close[i-1]) and (close[i] < open_[i-1]) and (open_[i] > close[i-1])
        
        if position == 0:
            # Long: Bullish engulfing in uptrend with volume confirmation
            if bullish_engulf and close[i] > ema21_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing in downtrend with volume confirmation
            elif bearish_engulf and close[i] < ema21_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bearish engulfing or trend reversal
            if bearish_engulf or close[i] < ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bullish engulfing or trend reversal
            if bullish_engulf or close[i] > ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals