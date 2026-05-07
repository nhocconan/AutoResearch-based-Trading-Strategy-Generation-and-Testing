#!/usr/bin/env python3
name = "1d_1w_RSI_Momentum_Trend_Filter"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI(14) for trend filter - only use completed weekly bars
    rsi_14 = 100 - (100 / (1 + (pd.Series(df_1w['close']).diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / 
                                     pd.Series(df_1w['close']).diff().clip(upper=0).abs().ewm(alpha=1/14, adjust=False).mean())))
    rsi_14 = rsi_14.fillna(50).values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    # Daily RSI(14) for entry signal
    rsi_daily = 100 - (100 / (1 + (pd.Series(close).diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / 
                                    pd.Series(close).diff().clip(upper=0).abs().ewm(alpha=1/14, adjust=False).mean())))
    rsi_daily = rsi_daily.fillna(50).values
    
    # Volume confirmation - 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(rsi_14_aligned[i]) or np.isnan(rsi_daily[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + weekly RSI > 50 (bullish trend) + volume spike
            if (rsi_daily[i] < 30 and 
                rsi_14_aligned[i] > 50 and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + weekly RSI < 50 (bearish trend) + volume spike
            elif (rsi_daily[i] > 70 and 
                  rsi_14_aligned[i] < 50 and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 50 (overbought) or trend changes
            if (rsi_daily[i] > 50 or 
                rsi_14_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 (oversold) or trend changes
            if (rsi_daily[i] < 50 or 
                rsi_14_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily RSI mean reversion with weekly trend filter and volume confirmation.
# In bear markets (2025+), RSI extremes often precede short-term reversals.
# The weekly RSI filter ensures we only trade counter-trend when the higher timeframe
# trend is still intact, reducing false signals during strong trends.
# Volume confirmation adds validity to reversal signals.
# Works in both bull and bear markets by adapting to the weekly trend direction.
# Position size 0.25 limits drawdown during adverse moves.