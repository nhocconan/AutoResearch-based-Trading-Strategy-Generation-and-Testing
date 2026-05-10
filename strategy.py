#!/usr/bin/env python3
# 6h_Adaptive_RSI_Bollinger_Bands
# Hypothesis: Combines adaptive RSI(3) with Bollinger Bands(20,2) for mean-reversion entries
# in ranging markets and trend-following exits in trending markets. Uses 1d ADX to filter
# regime: only trade when ADX < 25 (range) for mean reversion, or ADX > 25 (trend) for
# trend continuation. Designed for low trade frequency (target: 15-35 trades/year) with
# disciplined risk management via Bollinger Band middle band exits.
# Works in both bull and bear markets by adapting to regime conditions.

name = "6h_Adaptive_RSI_Bollinger_Bands"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Adaptive RSI(3) - more responsive than standard RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = pd.Series(gain).ewm(alpha=1/3, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/3, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands(20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = sma + (std_dev * bb_std)
    lower = sma - (std_dev * bb_std)
    
    # 1d ADX(14) for regime filtering
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed averages
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean()
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_ma / (tr_ma + 1e-10)
    di_minus = 100 * dm_minus_ma / (tr_ma + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(sma[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        rsi_val = rsi[i]
        close_val = close[i]
        bb_upper = upper[i]
        bb_lower = lower[i]
        bb_middle = sma[i]
        
        if position == 0:
            # Range market (ADX < 25): mean reversion at Bollinger Bands
            if adx_val < 25:
                if rsi_val < 30 and close_val <= bb_lower:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70 and close_val >= bb_upper:
                    signals[i] = -0.25
                    position = -1
            # Trending market (ADX >= 25): trend continuation with RSI pullback
            else:
                # Uptrend: buy on RSI pullback from overbought
                if rsi_val < 40 and close_val > bb_middle:
                    signals[i] = 0.25
                    position = 1
                # Downtrend: sell on RSI bounce from oversold
                elif rsi_val > 60 and close_val < bb_middle:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: RSI overbought or price touches upper band
            if rsi_val > 70 or close_val >= bb_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: RSI oversold or price touches lower band
            if rsi_val < 30 or close_val <= bb_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals