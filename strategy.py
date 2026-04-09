#!/usr/bin/env python3
# 6h_1w_1d_cci_trend_follow_v1
# Hypothesis: 6h strategy using weekly CCI trend filter and daily CCI pullback entries.
# Long: Weekly CCI(20) > 100 (uptrend) AND Daily CCI(14) < -100 (oversold pullback) AND 6h close > 6h open (bullish candle).
# Short: Weekly CCI(20) < -100 (downtrend) AND Daily CCI(14) > 100 (overbought pullback) AND 6h close < 6h open (bearish candle).
# Exit: Opposite CCI condition (weekly CCI crosses back through zero) or RSI(14) extreme (>80 for long, <20 for short).
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 15-30 trades/year.
# Works in bull markets via trend-following pullbacks and bear markets via shorting rallies in downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_cci_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # RSI(14) for emergency exit
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get 1d data for daily CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily CCI(14)
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    tp_s_1d = pd.Series(tp_1d)
    ma_1d = tp_s_1d.rolling(window=14, min_periods=14).mean()
    mad_1d = (tp_s_1d - ma_1d).abs().rolling(window=14, min_periods=14).mean()
    cci_1d = (tp_s_1d - ma_1d) / (0.015 * mad_1d)
    cci_1d_values = cci_1d.values
    
    # Align daily CCI to 6h
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d_values)
    
    # Get 1w data for weekly CCI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly CCI(20)
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    tp_s_1w = pd.Series(tp_1w)
    ma_1w = tp_s_1w.rolling(window=20, min_periods=20).mean()
    mad_1w = (tp_s_1w - ma_1w).abs().rolling(window=20, min_periods=20).mean()
    cci_1w = (tp_s_1w - ma_1w) / (0.015 * mad_1w)
    cci_1w_values = cci_1w.values
    
    # Align weekly CCI to 6h
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(cci_1w_aligned[i]) or
            np.isnan(rsi_values[i]) or
            np.isnan(close[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Bullish/bearish candle
        bullish_candle = close[i] > open_prices[i]
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Weekly CCI drops below zero (trend weakening) OR RSI overbought
            if (cci_1w_aligned[i] < 0 or rsi_values[i] > 80):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Weekly CCI rises above zero (trend weakening) OR RSI oversold
            if (cci_1w_aligned[i] > 0 or rsi_values[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Weekly uptrend AND Daily oversold pullback AND bullish candle
            if (cci_1w_aligned[i] > 100 and      # Weekly uptrend
                cci_1d_aligned[i] < -100 and     # Daily oversold pullback
                bullish_candle):                 # Bullish confirmation candle
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly downtrend AND Daily overbought pullback AND bearish candle
            elif (cci_1w_aligned[i] < -100 and   # Weekly downtrend
                  cci_1d_aligned[i] > 100 and    # Daily overbought pullback
                  bearish_candle):               # Bearish confirmation candle
                position = -1
                signals[i] = -0.25
    
    return signals