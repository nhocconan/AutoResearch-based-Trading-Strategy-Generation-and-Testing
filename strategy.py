#!/usr/bin/env python3
# 4h_Adaptive_Trend_Pullback_1dTrend_Volume
# Hypothesis: 4h trend-following with 1d EMA trend filter, volume confirmation, and pullback entries.
# Uses EMA alignment and RSI for pullback entries in trending markets. Works in bull (trend continuation)
# and bear (trend filter avoids false signals). Target: 20-40 trades/year.

name = "4h_Adaptive_Trend_Pullback_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h EMA for trend strength ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === RSI(14) for pullback entries ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Price above/below 4h EMA for trend strength
        price_above_ema = close[i] > ema_20[i]
        price_below_ema = close[i] < ema_20[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Pullback in uptrend - price touches 4h EMA with RSI oversold
            if (trend_up and price_below_ema and rsi[i] < 30 and vol_ok):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback in downtrend - price touches 4h EMA with RSI overbought
            elif (trend_down and price_above_ema and rsi[i] > 70 and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend reversal or overbought conditions
            if (not trend_up or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or oversold conditions
            if (not trend_down or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals