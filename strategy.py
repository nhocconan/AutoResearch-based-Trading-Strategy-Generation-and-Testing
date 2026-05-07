#!/usr/bin/env python3
# 4h_Engulfing_1dTrend_Volume
# Hypothesis: 4h chart strategy using bullish/bearish engulfing candle patterns filtered by 1d EMA50 trend and volume confirmation (1.5x average volume).
# Engulfing patterns signal strong reversal momentum. 1d EMA50 provides trend filter to avoid counter-trend trades.
# Volume confirms breakout validity. Designed to work in both bull and bear markets by filtering with trend and requiring volume confirmation.
# Target: 20-40 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "4h"
name = "4h_Engulfing_1dTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 1.5x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish engulfing: current green candle completely engulfs previous red candle
        bullish_engulf = (close[i] > open_prices[i]) and (open_prices[i-1] > close[i-1]) and \
                         (close[i] >= open_prices[i-1]) and (open_prices[i] <= close[i-1])
        # Bearish engulfing: current red candle completely engulfs previous green candle
        bearish_engulf = (close[i] < open_prices[i]) and (open_prices[i-1] < close[i-1]) and \
                         (open_prices[i] >= close[i-1]) and (close[i] <= open_prices[i-1])
        
        if position == 0:
            # Long: bullish engulfing with volume, and 1d trend is bullish (price > EMA50)
            if bullish_engulf and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing with volume, and 1d trend is bearish (price < EMA50)
            elif bearish_engulf and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish engulfing (reversal signal)
            if bearish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish engulfing (reversal signal)
            if bullish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals