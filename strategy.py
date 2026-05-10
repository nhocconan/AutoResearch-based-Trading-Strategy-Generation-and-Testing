#!/usr/bin/env python3
# 6H_Engulfing_Reversal_Volume
# Hypothesis: Engulfing candles on 6h with volume > 2x average signal reversals in both bull and bear markets.
# Works because institutional absorption creates sharp reversals after overextension.
# Volume confirms institutional participation, reducing false signals.
# Uses 1d trend filter: only trade against the daily trend (mean reversion in strong trends).
# Target: 20-40 trades/year per symbol.

name = "6H_Engulfing_Reversal_Volume"
timeframe = "6h"
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
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and \
                         (open_price[i-1] > close[i-1]) and \
                         (close[i] > open_price[i-1]) and \
                         (open_price[i] < close[i-1])
        
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and \
                         (open_price[i-1] < close[i-1]) and \
                         (open_price[i] > close[i-1]) and \
                         (close[i] < open_price[i-1])
        
        if position == 0:
            # Enter long: daily downtrend + bullish engulfing + volume
            if daily_down and bullish_engulf and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily uptrend + bearish engulfing + volume
            elif daily_up and bearish_engulf and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend resumes or opposite engulfing
            if not daily_down or bearish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend resumes or opposite engulfing
            if not daily_up or bullish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals