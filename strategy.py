#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h trend filter (EMA50) and 1d momentum filter (RSI<30 for long, RSI>70 for short).
Long when: price > 4h EMA50 AND 1d RSI < 30 AND 1h close > 1h open (bullish candle).
Short when: price < 4h EMA50 AND 1d RSI > 70 AND 1h close < 1h open (bearish candle).
Uses 4h for trend direction, 1d for extreme momentum, 1h for entry timing.
Session filter: 08-20 UTC to avoid low-liquidity hours.
Target: 20-60 trades/year to minimize fee drag. Uses discrete sizing 0.20.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for momentum (RSI14)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish/bearish 1h candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Long: uptrend (price > 4h EMA50) + extreme bearish momentum (1d RSI < 30) + bullish candle
            if (close[i] > ema_50_4h_aligned[i] and 
                rsi_14_1d_aligned[i] < 30 and 
                bullish_candle):
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < 4h EMA50) + extreme bullish momentum (1d RSI > 70) + bearish candle
            elif (close[i] < ema_50_4h_aligned[i] and 
                  rsi_14_1d_aligned[i] > 70 and 
                  bearish_candle):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal (price < 4h EMA50) or RSI normalizes (> 50)
            if (close[i] < ema_50_4h_aligned[i] or 
                rsi_14_1d_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend reversal (price > 4h EMA50) or RSI normalizes (< 50)
            if (close[i] > ema_50_4h_aligned[i] or 
                rsi_14_1d_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA50_1dRSI_Extreme"
timeframe = "1h"
leverage = 1.0