#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day 13-period EMA crossover with 1-week RSI filter and volume confirmation
# Long when EMA(13) crosses above EMA(34) on daily timeframe, RSI(1w) < 30 (oversold), and volume > 1.5x average
# Short when EMA(13) crosses below EMA(34) on daily timeframe, RSI(1w) > 70 (overbought), and volume > 1.5x average
# Daily EMA crossover provides timely trend changes, weekly RSI filters for extreme sentiment, volume confirms strength
# Works in bull markets by catching strong trends, in bear markets by fading overextended moves
# Target: 20-50 trades per year (80-200 over 4 years) with 0.25 position sizing

name = "4h_1dEMA13_34_1wRSI_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day EMA crossovers ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA(13) and EMA(34)
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # EMA crossover signal: 1 for bullish cross, -1 for bearish cross, 0 otherwise
    ema_cross = np.zeros(len(ema_13))
    ema_cross[1:] = np.where((ema_13[1:] > ema_34[1:]) & (ema_13[:-1] <= ema_34[:-1]), 1,
                            np.where((ema_13[1:] < ema_34[1:]) & (ema_13[:-1] >= ema_34[:-1]), -1, 0))
    
    # Align EMA crossover to 4h timeframe
    ema_cross_aligned = align_htf_to_ltf(prices, df_1d, ema_cross)
    
    # Calculate 1-week RSI for sentiment filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI(14)
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs)).values
    
    # Align weekly RSI to 4h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_cross_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: bullish EMA crossover, weekly RSI oversold (<30), volume confirmation
            if ema_cross_aligned[i] == 1 and rsi_1w_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish EMA crossover, weekly RSI overbought (>70), volume confirmation
            elif ema_cross_aligned[i] == -1 and rsi_1w_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish EMA crossover or weekly RSI overbought (>70)
            if ema_cross_aligned[i] == -1 or rsi_1w_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish EMA crossover or weekly RSI oversold (<30)
            if ema_cross_aligned[i] == 1 or rsi_1w_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals