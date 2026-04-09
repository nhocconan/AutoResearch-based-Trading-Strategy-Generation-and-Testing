#!/usr/bin/env python3
# 1d_ema_cross_volume_macd_v1
# Hypothesis: Daily EMA(21) cross with MACD histogram and volume confirmation captures major trend moves in both bull and bear markets.
# Uses EMA(9) and EMA(21) crossover, MACD(12,26,9) histogram > 0 for long/<0 for short, volume > 1.5x 20-day average.
# Includes volatility filter (ATR(14) < 50-day ATR mean) to avoid chop. Target: 15-25 trades/year (60-100 total).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_cross_volume_macd_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(9) and EMA(21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # MACD(12,26,9)
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - macd_signal
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # ATR(14) for volatility filter
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr14 = np.zeros(n)
    atr14[0] = tr[0]
    for i in range(1, n):
        atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # 50-day average ATR for volatility regime
    atr_ma_50 = np.zeros(n)
    atr_sum = 0
    for i in range(n):
        atr_sum += atr14[i]
        if i >= 50:
            atr_sum -= atr14[i-50]
        if i >= 49:
            atr_ma_50[i] = atr_sum / 50
    
    # Weekly trend filter: EMA(21) on weekly
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(macd_hist[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(atr14[i]) or np.isnan(atr_ma_50[i]) or \
           np.isnan(ema21_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid high volatility chop
        low_vol = atr14[i] < atr_ma_50[i] * 1.5
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema21_1w_aligned[i]
        weekly_downtrend = close[i] < ema21_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA cross down or MACD histogram negative
            if ema9[i] < ema21[i] or macd_hist[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up or MACD histogram positive
            if ema9[i] > ema21[i] or macd_hist[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: EMA cross up + MACD hist positive + volume + low vol + weekly uptrend
            if (ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and  # fresh cross
                macd_hist[i] > 0 and 
                volume[i] > vol_ma_20[i] * 1.5 and
                low_vol and
                weekly_uptrend):
                position = 1
                signals[i] = 0.25
            # Enter short: EMA cross down + MACD hist negative + volume + low vol + weekly downtrend
            elif (ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and  # fresh cross
                  macd_hist[i] < 0 and 
                  volume[i] > vol_ma_20[i] * 1.5 and
                  low_vol and
                  weekly_downtrend):
                position = -1
                signals[i] = -0.25
    
    return signals