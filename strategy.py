#!/usr/bin/env python3
name = "6h_LiquidityVoid_RSI_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily data for liquidity void detection and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Daily body size (close - open) to detect momentum direction
    open_1d = df_1d['open'].values
    body_size = close_1d - open_1d
    # Bullish body: positive and large; Bearish body: negative and large
    bullish_body = body_size > 0
    bearish_body = body_size < 0
    
    # Liquidity void detection: gaps between daily candles
    # Bullish void: today's low > yesterday's high (gap up)
    # Bearish void: today's high < yesterday's low (gap down)
    bullish_void = low_1d > np.concatenate([[high_1d[0]], high_1d[:-1]])
    bearish_void = high_1d < np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    # Align all to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bullish_void_aligned = align_htf_to_ltf(prices, df_1d, bullish_void.astype(float))
    bearish_void_aligned = align_htf_to_ltf(prices, df_1d, bearish_void.astype(float))
    bullish_body_aligned = align_htf_to_ltf(prices, df_1d, bullish_body.astype(float))
    bearish_body_aligned = align_htf_to_ltf(prices, df_1d, bearish_body.astype(float))
    
    # Volume spike on 6h: current volume > 2.0x 20-period average (strict)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(bullish_void_aligned[i]) or 
            np.isnan(bearish_void_aligned[i]) or np.isnan(bullish_body_aligned[i]) or
            np.isnan(bearish_body_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish liquidity void + bullish daily body + RSI > 55 + volume spike
            if (bullish_void_aligned[i] > 0.5 and 
                bullish_body_aligned[i] > 0.5 and 
                rsi_1d_aligned[i] > 55 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish liquidity void + bearish daily body + RSI < 45 + volume spike
            elif (bearish_void_aligned[i] > 0.5 and 
                  bearish_body_aligned[i] > 0.5 and 
                  rsi_1d_aligned[i] < 45 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish void appears or RSI < 40
            if bearish_void_aligned[i] > 0.5 or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish void appears or RSI > 60
            if bullish_void_aligned[i] > 0.5 or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals