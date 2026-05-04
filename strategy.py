#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF for signal direction and 1h for entry timing.
# Long when: price > 4h EMA50, price > 1d EMA200, and 1h RSI(14) crosses above 30 (mean reversion in uptrend).
# Short when: price < 4h EMA50, price < 1d EMA200, and 1h RSI(14) crosses below 70 (mean reversion in downtrend).
# Uses 4h EMA50 for intermediate trend and 1d EMA200 for long-term trend alignment.
# RSI extremes provide mean-reversion entries within the trend, reducing whipsaw.
# Session filter (08-20 UTC) to avoid low-volume noise periods.
# Target: 15-30 trades/year per symbol via strict trend+RSI confluence.

name = "1h_EMA50_EMA200_RSI_MeanReversion_TrendFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Get 1d data for EMA200 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # RSI cross signals
    rsi_above_30 = (rsi > 30) & (np.append([False], rsi[:-1] <= 30))  # Cross above 30
    rsi_below_70 = (rsi < 70) & (np.append([False], rsi[:-1] >= 70))  # Cross below 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(ema_1d_200_aligned[i]) or 
            np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: uptrend alignment + RSI mean reversion entry
            if (close[i] > ema_4h_50_aligned[i] and 
                close[i] > ema_1d_200_aligned[i] and 
                rsi_above_30[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: downtrend alignment + RSI mean reversion entry
            elif (close[i] < ema_4h_50_aligned[i] and 
                  close[i] < ema_1d_200_aligned[i] and 
                  rsi_below_70[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: trend breakdown or RSI overbought
            if (close[i] < ema_4h_50_aligned[i] or 
                close[i] < ema_1d_200_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: trend reversal or RSI oversold
            if (close[i] > ema_4h_50_aligned[i] or 
                close[i] > ema_1d_200_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals