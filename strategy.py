#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Financial_Freedom_Momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for momentum filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily RSI(14) for momentum filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = np.concatenate([[np.nan] * 14, rsi_14_1d[14:]])
    
    # Daily close for EMA filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h ATR(14) for volatility filter and stop
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA(20) for trend
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_1d = rsi_14_1d_aligned[i]
        ema_50_d = ema_50_1d_aligned[i]
        ema_20_4h = ema_20[i]
        atr = atr_14[i]
        
        # Momentum condition: RSI > 50 and price above daily EMA50
        momentum_long = rsi_1d > 50 and price > ema_50_d
        momentum_short = rsi_1d < 50 and price < ema_50_d
        
        # Trend condition: 4h price above/below EMA20
        trend_up = price > ema_20_4h
        trend_down = price < ema_20_4h
        
        if position == 0:
            # Long: momentum + trend alignment
            if momentum_long and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: momentum + trend alignment
            elif momentum_short and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: momentum reversal or trend break or ATR stop
            if (not momentum_long) or (not trend_up) or (price < ema_20_4h - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: momentum reversal or trend break or ATR stop
            if (not momentum_short) or (not trend_down) or (price > ema_20_4h + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals