#!/usr/bin/env python3
# 4h_PriceAction_Trend_Confirmation
# Hypothesis: Price action confirmation with multi-timeframe trend and volume filters.
# Uses candlestick patterns (engulfing) for entry, confirmed by 1d EMA trend and volume spike.
# Works in bull/bear: Trend filter prevents counter-trend trades, volume confirms momentum.
# Designed for low trade frequency (<50/year) to minimize fee drag.

name = "4h_PriceAction_Trend_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (ema_200_1d[i-1] * 199 + close_1d[i]) / 200
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > open_price[1]) & (open_price < close[1]) & \
                        (close - open_price > open_price[1] - close[1])
    
    # Bearish engulfing: current candle engulfs previous bullish candle
    bearish_engulfing = (close < open_price) & (open_price > close) & \
                        (open_price > close[1]) & (close < open_price[1]) & \
                        (open_price - close > close[1] - open_price[1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish engulfing AND uptrend (close > EMA200) AND volume spike
            if (bullish_engulfing[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish engulfing AND downtrend (close < EMA200) AND volume spike
            elif (bearish_engulfing[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish engulfing OR trend reversal (close < EMA200)
            if bearish_engulfing[i] or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish engulfing OR trend reversal (close > EMA200)
            if bullish_engulfing[i] or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals