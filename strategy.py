#!/usr/bin/env python3
# Hypothesis: 6h MACD histogram with 12h trend filter and volume confirmation
# Long when: MACD histogram > 0 (bullish momentum), 12h EMA50 uptrend (close > EMA50), volume > 1.5x 20-period average
# Short when: MACD histogram < 0 (bearish momentum), 12h EMA50 downtrend (close < EMA50), volume > 1.5x 20-period average
# Exit when: MACD histogram crosses zero OR 12h trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 25-50 trades/year.
# Designed to work in both bull (MACD + trend + volume) and bear (MACD + trend + volume) markets.

name = "6h_MACD_12hTrend_Volume"
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
    volume = prices['volume'].values
    
    # Calculate MACD
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - signal_line
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(macd_hist.iloc[i]) if hasattr(macd_hist, 'iloc') else np.isnan(macd_hist[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get MACD histogram value
        macd_hist_val = macd_hist.iloc[i] if hasattr(macd_hist, 'iloc') else macd_hist[i]
        
        if position == 0:
            # Enter long: MACD histogram > 0, 12h close > EMA50 (uptrend), volume spike
            if (macd_hist_val > 0 and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: MACD histogram < 0, 12h close < EMA50 (downtrend), volume spike
            elif (macd_hist_val < 0 and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: MACD histogram crosses below zero OR 12h trend turns down (close < EMA50)
            if (macd_hist_val <= 0) or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: MACD histogram crosses above zero OR 12h trend turns up (close > EMA50)
            if (macd_hist_val >= 0) or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals