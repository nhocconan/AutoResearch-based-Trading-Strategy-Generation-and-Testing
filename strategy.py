#!/usr/bin/env python3
# 4H_MACD_RSI_CONFLUENCE_1D_TREND_FILTER
# Hypothesis: MACD(12,26,9) + RSI(14) confluence with 1d EMA50 trend filter.
# In 1d uptrend: long when MACD line crosses above signal line and RSI > 50.
# In 1d downtrend: short when MACD line crosses below signal line and RSI < 50.
# Uses volume confirmation (>1.5x 20-period average) to filter false breakouts.
# Works in both bull and bear markets: trend filter avoids counter-trend trades,
# MACD/RSI captures momentum within trend. Target: 20-30 trades/year on 4h timeframe.

name = "4H_MACD_RSI_CONFLUENCE_1D_TREND_FILTER"
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
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # MACD on price
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = (ema12 - ema26).values
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(macd_line[i]) or 
            np.isnan(signal_line[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + MACD bullish cross + RSI > 50 + volume confirmation
            if (close[i] > ema50_1d_aligned[i] and 
                macd_line[i] > signal_line[i] and 
                macd_line[i-1] <= signal_line[i-1] and  # bullish cross
                rsi[i] > 50 and 
                vol_conf[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + MACD bearish cross + RSI < 50 + volume confirmation
            elif (close[i] < ema50_1d_aligned[i] and 
                  macd_line[i] < signal_line[i] and 
                  macd_line[i-1] >= signal_line[i-1] and  # bearish cross
                  rsi[i] < 50 and 
                  vol_conf[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or MACD bearish cross
            if (close[i] <= ema50_1d_aligned[i] or 
                (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or MACD bullish cross
            if (close[i] >= ema50_1d_aligned[i] or 
                (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals