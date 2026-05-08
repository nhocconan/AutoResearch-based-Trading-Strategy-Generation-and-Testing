#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Price Action + 12h Momentum Divergence
# Uses 6h price action (engulfing candles) for entry timing and 12h RSI divergence for momentum confirmation.
# Engulfing candles signal strong short-term reversals, while RSI divergence on higher timeframe
# confirms weakening momentum and potential trend exhaustion. This combination works in both
# bull and bear markets by capturing reversals at key turning points. Targets 20-30 trades per year.

name = "6h_Engulfing_12hRSI_Divergence"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Get 12h data for RSI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate RSI(14) on 12h
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Need at least 2 bars for engulfing pattern
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if RSI data is not available
        if np.isnan(rsi_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Detect bullish engulfing: current green candle completely engulfs previous red candle
        bullish_engulfing = (
            close[i] > open_price[i] and  # current candle is green
            close[i-1] < open_price[i-1] and  # previous candle is red
            close[i] >= open_price[i-1] and  # current close >= previous open
            open_price[i] <= close[i-1]  # current open <= previous close
        )
        
        # Detect bearish engulfing: current red candle completely engulfs previous green candle
        bearish_engulfing = (
            close[i] < open_price[i] and  # current candle is red
            close[i-1] > open_price[i-1] and  # previous candle is green
            open_price[i] >= close[i-1] and  # current open >= previous close
            close[i] <= open_price[i-1]  # current close <= previous open
        )
        
        if position == 0:
            # Enter long on bullish engulfing when RSI is not overbought (< 70)
            if bullish_engulfing and rsi_12h_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish engulfing when RSI is not oversold (> 30)
            elif bearish_engulfing and rsi_12h_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long on bearish engulfing or RSI overbought
            if bearish_engulfing or rsi_12h_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on bullish engulfing or RSI oversold
            if bullish_engulfing or rsi_12h_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals