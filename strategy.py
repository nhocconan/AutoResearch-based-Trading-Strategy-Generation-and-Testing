#!/usr/bin/env python3
# 4h_Engulfing_Candle_With_Volume_Filter
# Hypothesis: Bullish/bearish engulfing candles on 4h timeframe with volume confirmation (>2x average) and 12h EMA trend filter.
# Works in bull/bear markets: Engulfing candles signal strong momentum reversals, volume confirms institutional participation,
# and the 12h EMA filter ensures trades align with higher timeframe trend to avoid counter-trend whipsaws.
# Designed for low trade frequency (<40/year) to minimize fee drag while capturing high-probability moves.

name = "4h_Engulfing_Candle_With_Volume_Filter"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Detect bullish and bearish engulfing candles
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulf: current green candle engulfs previous red candle
        if (close[i] > open_price[i] and  # current candle is green
            open_price[i-1] > close[i-1] and  # previous candle is red
            open_price[i] <= close[i-1] and  # current open <= previous close
            close[i] >= open_price[i-1]):  # current close >= previous open
            bullish_engulf[i] = True
        
        # Bearish engulf: current red candle engulfs previous green candle
        elif (close[i] < open_price[i] and  # current candle is red
              open_price[i-1] < close[i-1] and  # previous candle is green
              open_price[i] >= close[i-1] and  # current open >= previous close
              close[i] <= open_price[i-1]):  # current close <= previous open
            bearish_engulf[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish engulfing candle + uptrend (price > EMA50) + volume spike
            if (bullish_engulf[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish engulfing candle + downtrend (price < EMA50) + volume spike
            elif (bearish_engulf[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish engulfing candle OR trend reversal (price < EMA50)
            if bearish_engulf[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish engulfing candle OR trend reversal (price > EMA50)
            if bullish_engulf[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals