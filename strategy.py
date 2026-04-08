#!/usr/bin/env python3
# 4h_1d_rsi_engulfing_v1
# Hypothesis: RSI extremes with bullish/bearish engulfing candles on 4h, filtered by 1-day EMA trend.
# Long: RSI(14) < 30 AND bullish engulfing candle AND price > 1-day EMA200.
# Short: RSI(14) > 70 AND bearish engulfing candle AND price < 1-day EMA200.
# Exit: RSI crosses back above 50 (long) or below 50 (short).
# Designed to capture mean reversals in both bull and bear markets with strict entry criteria.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_engulfing_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # 4h Engulfing patterns
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulfing: current green candle engulfs previous red candle
        if (close[i] > open_price[i] and  # current bullish
            open_price[i] < close[i-1] and  # current open < previous close
            close[i] > open_price[i-1] and  # current close > previous open
            open_price[i-1] > close[i-1]):  # previous bearish
            bullish_engulf[i] = True
        
        # Bearish engulfing: current red candle engulfs previous green candle
        if (close[i] < open_price[i] and  # current bearish
            open_price[i] > close[i-1] and  # current open > previous close
            close[i] < open_price[i-1] and  # current close < previous open
            open_price[i-1] < close[i-1]):  # previous bullish
            bearish_engulf[i] = True
    
    # 1-day EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_1d_200[i] = close_1d[i] * (2/201) + ema_1d_200[i-1] * (199/201)
    
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        price = close[i]
        rsi_val = rsi[i]
        bull_eng = bullish_engulf[i]
        bear_eng = bearish_engulf[i]
        ema_1d = ema_1d_200_aligned[i]
        
        if np.isnan(ema_1d):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if rsi_val > 50:  # Exit when RSI crosses back above 50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if rsi_val < 50:  # Exit when RSI crosses back below 50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if rsi_val < 30 and bull_eng and price > ema_1d:
                position = 1
                signals[i] = 0.25
            elif rsi_val > 70 and bear_eng and price < ema_1d:
                position = -1
                signals[i] = -0.25
    
    return signals