#!/usr/bin/env python3
# 4H_1W_RSI_EXTREMES_VOLUME_CONFIRMATION
# Hypothesis: RSI(14) extremes (<30 or >70) on 4h with weekly trend filter and volume confirmation captures mean-reversion in overbought/oversold conditions while avoiding counter-trend trades. Weekly trend ensures alignment with higher timeframe momentum, volume reduces false signals. Designed for 20-40 trades/year to minimize fee drag.

name = "4H_1W_RSI_EXTREMES_VOLUME_CONFIRMATION"
timeframe = "4h"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 4h RSI(14)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Weekly trend: close vs EMA21
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_1w_up = close_1w > ema21_1w
    trend_1w_down = close_1w < ema21_1w
    
    # Align weekly trend to 4h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(trend_1w_up_aligned[i]) or 
            np.isnan(trend_1w_down_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: RSI oversold (<30) with weekly uptrend and volume
            if (rsi[i] < 30 and trend_1w_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) with weekly downtrend and volume
            elif (rsi[i] > 70 and trend_1w_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when RSI returns to neutral (50) or trend fails
            if (rsi[i] >= 50 or trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when RSI returns to neutral (50) or trend fails
            if (rsi[i] <= 50 or trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals