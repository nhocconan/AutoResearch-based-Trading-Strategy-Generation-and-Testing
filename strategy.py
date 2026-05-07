#!/usr/bin/env python3
name = "1d_Keltner_Squeeze_Momentum_v1"
timeframe = "1d"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d ATR for Keltner Channels and volatility
    def calculate_atr(high, low, close, period=14):
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # 1d Keltner Channel (20, 2)
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper_1d = ema20_1d + 2 * atr_1d
    kc_lower_1d = ema20_1d - 2 * atr_1d
    
    # 1d Bollinger Bands (20, 2) for squeeze detection
    sma20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = sma20_1d + 2 * std20_1d
    bb_lower_1d = sma20_1d - 2 * std20_1d
    
    # Squeeze condition: Bollinger Bands inside Keltner Channels
    squeeze_condition = (bb_upper_1d <= kc_upper_1d) & (bb_lower_1d >= kc_lower_1d)
    
    # 1d momentum: RSI(14) for overbought/oversold
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(df_1d['close'].values, 14)
    
    # 1w trend filter: EMA50 slope
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope_1w = np.diff(ema50_1w, prepend=0)
    
    # Align indicators to 1d timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope_1w)
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper_1d)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower_1d)
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(squeeze_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema50_slope_aligned[i]) or np.isnan(kc_upper_aligned[i]) or 
            np.isnan(kc_lower_aligned[i]) or np.isnan(ema20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Squeeze breakout above KC upper with bullish weekly trend and RSI not overbought
            if (close[i] > kc_upper_aligned[i] and squeeze_aligned[i-1] and 
                ema50_slope_aligned[i] > 0 and rsi_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: Squeeze breakout below KC lower with bearish weekly trend and RSI not oversold
            elif (close[i] < kc_lower_aligned[i] and squeeze_aligned[i-1] and 
                  ema50_slope_aligned[i] < 0 and rsi_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below KC lower or squeeze fires (low volatility)
            if close[i] < kc_lower_aligned[i] or squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above KC upper or squeeze fires (low volatility)
            if close[i] > kc_upper_aligned[i] or squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals