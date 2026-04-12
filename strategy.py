#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_rsi_momentum_v1
# Uses weekly RSI to determine trend strength and daily RSI for mean-reversion entries.
# Long when weekly RSI > 50 (bullish regime) and daily RSI < 30 (oversold pullback).
# Short when weekly RSI < 50 (bearish regime) and daily RSI > 70 (overbought rally).
# Includes volume confirmation to filter false signals and ATR-based volatility filter.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).

name = "1d_1w_rsi_momentum_v1"
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
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI (14) for trend regime
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_1w = np.where(avg_loss == 0, 100, rsi_14)
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily RSI (14) for entry signals
    delta_d = np.diff(close, prepend=close[0])
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    avg_gain_d = pd.Series(gain_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_d = pd.Series(loss_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_d = avg_gain_d / (avg_loss_d + 1e-10)
    rsi_14_d = 100 - (100 / (1 + rs_d))
    rsi_1d = np.where(avg_loss_d == 0, 100, rsi_14_d)
    
    # Volume confirmation: volume > 1.3 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # ATR-based volatility filter (avoid choppy markets)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ratio = atr / (pd.Series(atr).rolling(window=50, min_periods=50).mean().values + 1e-10)
    vol_filter = atr_ratio < 1.5  # Avoid excessively volatile periods
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(vol_confirm[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: weekly RSI > 50 (bullish regime) AND daily RSI < 30 (oversold)
        if (rsi_1w_aligned[i] > 50 and rsi_1d[i] < 30 and 
            vol_confirm[i] and vol_filter[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: weekly RSI < 50 (bearish regime) AND daily RSI > 70 (overbought)
        elif (rsi_1w_aligned[i] < 50 and rsi_1d[i] > 70 and 
              vol_confirm[i] and vol_filter[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite weekly RSI regime or RSI mean reversion
        elif ((rsi_1w_aligned[i] < 50 and position == 1) or  # regime change to bearish
              (rsi_1w_aligned[i] > 50 and position == -1) or  # regime change to bullish
              (rsi_1d[i] > 70 and position == 1) or          # overbought exit long
              (rsi_1d[i] < 30 and position == -1)):          # oversold exit short
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals