#!/usr/bin/env python3
name = "4h_RSI_Trend_Squeeze_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Trend Filter (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- RSI(14) on 4h ---
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # --- Bollinger Band Squeeze (volatility filter) ---
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_mean = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < 0.8 * bb_width_mean  # low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(squeeze[i]) or
            np.isnan(bb_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 50 (momentum) + price > 1d EMA50 (trend) + Bollinger squeeze (low vol breakout setup)
            if (rsi[i] > 50 and 
                close[i] > ema50_1d_aligned[i] and 
                squeeze[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 (momentum) + price < 1d EMA50 (trend) + Bollinger squeeze (low vol breakdown setup)
            elif (rsi[i] < 50 and 
                  close[i] < ema50_1d_aligned[i] and 
                  squeeze[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI < 40 (loss of momentum) OR price < 1d EMA50 (trend break)
            if rsi[i] < 40 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI > 60 (loss of momentum) OR price > 1d EMA50 (trend break)
            if rsi[i] > 60 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals