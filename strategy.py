#!/usr/bin/env python3
# 12h_RSI_Pullback_WeeklyTrend_Volume
# Hypothesis: In 12h timeframe, enter long when RSI(14) pulls back to 40-50 during weekly uptrend with volume confirmation, short when RSI pulls back to 50-60 during weekly downtrend. Uses weekly trend filter to avoid counter-trend trades, targeting 20-40 trades/year for low friction in both bull and bear markets.

name = "12h_RSI_Pullback_WeeklyTrend_Volume"
timeframe = "12h"
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
    
    # === Weekly trend filter: EMA50 on weekly close ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === RSI(14) on 12h close ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation: 24-period average ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: weekly EMA50 direction
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # RSI pullback zones
        rsi_long_zone = (rsi[i] >= 40) & (rsi[i] <= 50)
        rsi_short_zone = (rsi[i] >= 50) & (rsi[i] <= 60)
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: RSI pullback to 40-50 in weekly uptrend with volume
            if rsi_long_zone and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI pullback to 50-60 in weekly downtrend with volume
            elif rsi_short_zone and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI > 55 or trend reversal
            if rsi[i] > 55 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 45 or trend reversal
            if rsi[i] < 45 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals