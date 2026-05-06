#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly EMA50 trend filter with daily RSI(14) mean reversion
# Weekly EMA50 determines long-term trend (bull/bear), daily RSI provides entry timing
# In bull markets (price > weekly EMA50): long when RSI < 30 (oversold pullback)
# In bear markets (price < weekly EMA50): short when RSI > 70 (overbought rally)
# Uses volume confirmation (>1.5x 20-period average) to filter false signals
# Designed for low turnover: ~20-40 trades/year with 0.25 position sizing
# Works in both bull/bear markets by adapting to trend context

name = "6h_WeeklyEMA50_DailyRSI_MeanReversion_v1"
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
    
    # Calculate weekly EMA50 ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily RSI(14) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily RSI(14)
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: bullish trend (price > weekly EMA50) + oversold RSI + volume
            if close[i] > ema50_1w_aligned[i] and rsi_1d_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: bearish trend (price < weekly EMA50) + overbought RSI + volume
            elif close[i] < ema50_1w_aligned[i] and rsi_1d_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish trend reversal or RSI overbought
            if close[i] < ema50_1w_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish trend reversal or RSI oversold
            if close[i] > ema50_1w_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals