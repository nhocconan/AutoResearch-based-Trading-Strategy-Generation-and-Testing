#!/usr/bin/env python3
"""
1d_EngulfingPattern_1wTrend_VolumeConfirm
Hypothesis: Bullish/Bearish engulfing patterns on daily chart with weekly trend filter and volume confirmation.
Engulfing patterns signal strong reversals. Weekly trend filter ensures trading with higher timeframe momentum.
Volume confirmation (>1.5x 20-day average) validates pattern strength. Designed for low trade frequency (10-25/year)
to minimize fee drift. Works in both bull and bear markets by following weekly trend.
"""

name = "1d_EngulfingPattern_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (ema_20_1w[i-1] * 19 + close_1w[i]) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume / 20-day average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(20, 20)  # Ensure volume MA and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Bullish engulfing: current candle engulfs previous bearish candle
            bullish_engulf = (close[i] > open_price[i-1] and 
                             open_price[i] < close[i-1] and 
                             close[i-1] < open_price[i-1])  # previous candle bearish
            
            # Bearish engulfing: current candle engulfs previous bullish candle
            bearish_engulf = (open_price[i] > close[i-1] and 
                             close[i] < open_price[i-1] and 
                             close[i-1] > open_price[i-1])  # previous candle bullish
            
            # Enter long: bullish engulfing AND uptrend (close > weekly EMA20) AND volume confirmation
            if (bullish_engulf and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: bearish engulfing AND downtrend (close < weekly EMA20) AND volume confirmation
            elif (bearish_engulf and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: bearish engulfing OR trend reversal (close < weekly EMA20)
                bearish_engulf = (open_price[i] > close[i-1] and 
                                 close[i] < open_price[i-1] and 
                                 close[i-1] > open_price[i-1])
                if bearish_engulf or close[i] < ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: bullish engulfing OR trend reversal (close > weekly EMA20)
                bullish_engulf = (close[i] > open_price[i-1] and 
                                 open_price[i] < close[i-1] and 
                                 close[i-1] < open_price[i-1])
                if bullish_engulf or close[i] > ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals