#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h chart with 4h RSI(14) and 1d EMA(200) for trend filter.
# Long when 4h RSI < 30 (oversold) and price > 1d EMA200 (uptrend bias).
# Short when 4h RSI > 70 (overbought) and price < 1d EMA200 (downtrend bias).
# Entry only during 08-20 UTC session to avoid low-liquidity hours.
# Uses volume confirmation (1h volume > 1.5x 20-bar average) to filter false signals.
# Fixed position size 0.20 to control risk and minimize fee churn.
# Target: 15-30 trades/year per symbol (~60-120 over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for RSI (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # RSI(14) calculation
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Load 1d data for EMA200 (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: 1h volume > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if NaN in critical values
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip outside session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi = rsi_4h_aligned[i]
        ema200 = ema_200_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: oversold RSI in uptrend bias
            if rsi < 30 and price > ema200 and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: overbought RSI in downtrend bias
            elif rsi > 70 and price < ema200 and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or trend bias fails
            if rsi > 50 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral or trend bias fails
            if rsi < 50 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_RSI14_1d_EMA200_SessionVolFilter_v1"
timeframe = "1h"
leverage = 1.0