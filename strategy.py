#!/usr/bin/env python3
# 12h_Engulfing_Candle_1dTrend
# Hypothesis: Use bullish/bearish engulfing candles on 12h for entry, filtered by 1d EMA trend direction.
# Go long on bullish engulfing when price > 1d EMA50, short on bearish engulfing when price < 1d EMA50.
# Exit on opposite engulfing signal. Designed for low frequency (<30 trades/year) to avoid fee drag.
# Engulfing candles signal strong momentum shifts, effective in both bull and bear markets when aligned with higher timeframe trend.

name = "12h_Engulfing_Candle_1dTrend"
timeframe = "12h"
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 2 candles for engulfing pattern
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Bullish engulfing: current green candle completely engulfs previous red candle
        bullish_engulfing = (
            close[i] > open_price[i] and  # current candle is green
            open_price[i-1] > close[i-1] and  # previous candle is red
            close[i] >= open_price[i-1] and  # current close >= previous open
            open_price[i] <= close[i-1]  # current open <= previous close
        )
        
        # Bearish engulfing: current red candle completely engulfs previous green candle
        bearish_engulfing = (
            close[i] < open_price[i] and  # current candle is red
            open_price[i-1] < close[i-1] and  # previous candle is green
            close[i] <= open_price[i-1] and  # current close <= previous open
            open_price[i] >= close[i-1]  # current open >= previous close
        )
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # LONG: Bullish engulfing AND price above 1d EMA50
            if bullish_engulfing and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish engulfing AND price below 1d EMA50
            elif bearish_engulfing and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bearish engulfing signal
            if bearish_engulfing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish engulfing signal
            if bullish_engulfing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals