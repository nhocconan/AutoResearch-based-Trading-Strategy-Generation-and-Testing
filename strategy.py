#!/usr/bin/env python3
"""
6h_1d_ADX_Trend_with_RSI_Momentum_Exit
Hypothesis: Uses 1d ADX > 25 to identify strong trends, enters on 6h pullbacks to EMA21 in trend direction with RSI momentum confirmation.
Exits when RSI shows exhaustion (overbought/oversold) or trend weakens (ADX < 20). Works in bull/bear by following 1d trend.
Targets low trade frequency (15-30/year) via strong trend filter and momentum-based exits.
"""

name = "6h_1d_ADX_Trend_with_RSI_Momentum_Exit"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(np.abs(high[1:] - low[:-1]), np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[:-1] - close[:-1])))
    
    # Pad to original length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx

def calculate_ema(arr, period):
    """Calculate EMA"""
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first value
    rsi = np.concatenate([[50], rsi])
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d ADX for Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # --- 6h Indicators for Entry/Exit ---
    ema21_6h = calculate_ema(close, 21)
    rsi_6h = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_6h[i]) or np.isnan(ema21_6h[i]) or np.isnan(rsi_6h[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend strength filters
        strong_trend = adx_1d_6h[i] > 25
        weak_trend = adx_1d_6h[i] < 20
        
        if position == 0:
            # Long: strong uptrend + price near EMA21 + RSI momentum
            if (strong_trend and 
                close[i] > ema21_6h[i] and 
                close[i-1] <= ema21_6h[i-1] and  # crossed above EMA21 this bar
                rsi_6h[i] > 50 and rsi_6h[i] < 70):  # momentum but not overbought
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend + price near EMA21 + RSI momentum
            elif (strong_trend and 
                  close[i] < ema21_6h[i] and 
                  close[i-1] >= ema21_6h[i-1] and  # crossed below EMA21 this bar
                  rsi_6h[i] < 50 and rsi_6h[i] > 30):  # momentum but not oversold
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend weakens OR RSI overbought
                if weak_trend or rsi_6h[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend weakens OR RSI oversold
                if weak_trend or rsi_6h[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals