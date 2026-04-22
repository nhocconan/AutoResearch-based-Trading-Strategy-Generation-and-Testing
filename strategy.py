#!/usr/bin/env python3

"""
Hypothesis: 1-hour trend-following strategy using 4-hour EMA trend filter with 1-hour RSI pullback entries.
Trades pullbacks to the 4-hour EMA during strong trends, using RSI(14) < 30 for longs and > 70 for shorts.
Uses session filter (08-20 UTC) to avoid low-liquidity hours. Designed for 15-30 trades/year to minimize fee drag.
Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4-hour EMA for trend filter (21-period)
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1-hour RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold in uptrend (price above 4h EMA)
            if rsi[i] < 30 and close[i] > ema_21_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought in downtrend (price below 4h EMA)
            elif rsi[i] > 70 and close[i] < ema_21_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI > 50 or price closes below 4h EMA
                if rsi[i] > 50 or close[i] < ema_21_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI < 50 or price closes above 4h EMA
                if rsi[i] < 50 or close[i] > ema_21_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA21_RSI_Pullback_Session"
timeframe = "1h"
leverage = 1.0