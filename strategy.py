#!/usr/bin/env python3
# 1h_4H_1D_Combo_RSI_Divergence
# Hypothesis: Use 4h for trend direction (EMA50), 1d for regime (ADX), and 1h for entry timing via RSI divergence.
# Long when 4h EMA50 up, 1d ADX < 25 (range), and bullish RSI divergence on 1h.
# Short when 4h EMA50 down, 1d ADX < 25, and bearish RSI divergence on 1h.
# Exit on 4h EMA50 crossover or ADX > 30 (trend regime).
# Designed for low trade frequency and works in both bull/bear via regime adaptation.

name = "1h_4H_1D_Combo_RSI_Divergence"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, period=14):
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema4h_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema4h_50)
    
    # Load 1d data for regime (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1h RSI for divergence
    rsi_1h = calculate_rsi(close, 14)
    
    # Volume filter: 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema4h_50_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(rsi_1h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema4h_trend = ema4h_50_aligned[i]
        adx_regime = adx_1d_aligned[i]
        rsi_now = rsi_1h[i]
        vol_now = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Detect RSI divergence (simplified: price making new low/high, RSI not)
        bullish_div = False
        bearish_div = False
        if i >= 5:
            # Bullish divergence: price lower low, RSI higher low
            if low[i] < low[i-5] and rsi_now > rsi_1h[i-5]:
                bullish_div = True
            # Bearish divergence: price higher high, RSI lower high
            if high[i] > high[i-5] and rsi_now < rsi_1h[i-5]:
                bearish_div = True
        
        if position == 0:
            # Only trade in ranging regime (ADX < 25) with volume confirmation
            if adx_regime < 25 and vol_now > vol_ma_val:
                # LONG: 4h uptrend + bullish RSI divergence
                if close[i] > ema4h_trend and bullish_div:
                    signals[i] = 0.20
                    position = 1
                # SHORT: 4h downtrend + bearish RSI divergence
                elif close[i] < ema4h_trend and bearish_div:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h downtrend OR ADX > 30 (trending regime)
            if close[i] < ema4h_trend or adx_regime > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h uptrend OR ADX > 30
            if close[i] > ema4h_trend or adx_regime > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals