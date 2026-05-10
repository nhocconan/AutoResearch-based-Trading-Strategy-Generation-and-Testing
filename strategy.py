#!/usr/bin/env python3
# 4h_HTF_Bullish_Engulfing_Pattern_Volume
# Hypothesis: Bullish engulfing candlestick patterns on the 4h chart, when occurring 
# above the 12h EMA50 (trend filter) and with volume confirmation, capture momentum 
# in both bull and bear markets. The pattern indicates strong buying pressure overcoming 
# prior selling, and the 12h EMA filter ensures alignment with higher timeframe trend. 
# Volume confirmation filters weak signals. Engulfing patterns are less frequent than 
# simple breakouts, naturally limiting trade frequency to avoid fee drag.

name = "4h_HTF_Bullish_Engulfing_Pattern_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish engulfing: current bullish candle engulfs previous bearish candle
        bullish_engulfing = (
            close[i] > open_price[i] and  # current candle bullish
            open_price[i-1] > close[i-1] and  # previous candle bearish
            close[i] > open_price[i-1] and  # current close > previous open
            open_price[i] < close[i-1]  # current open < previous close
        )
        
        # Bearish engulfing: current bearish candle engulfs previous bullish candle
        bearish_engulfing = (
            close[i] < open_price[i] and  # current candle bearish
            open_price[i-1] < close[i-1] and  # previous candle bullish
            close[i] < open_price[i-1] and  # current close < previous open
            open_price[i] > close[i-1]  # current open > previous close
        )
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: bullish engulfing + uptrend + volume
            if bullish_engulfing and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish engulfing + downtrend + volume
            elif bearish_engulfing and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish engulfing or trend breaks
            if bearish_engulfing or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish engulfing or trend breaks
            if bullish_engulfing or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals