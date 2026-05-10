#!/usr/bin/env python3
# 1d_7Day_RSI_MeanReversion_WeeklyTrend_Filter
# Hypothesis: Weekly RSI(7) mean reversion with daily trend filter. In uptrend (price > weekly EMA), buy when weekly RSI < 30. In downtrend (price < weekly EMA), sell when weekly RSI > 70. Uses weekly timeframe for signal generation, daily for trend alignment to avoid counter-trend trades. Targets 10-25 trades/year to minimize fee drag while capturing mean reversion in both bull and bear markets.

name = "1d_7Day_RSI_MeanReversion_WeeklyTrend_Filter"
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
    volume = prices['volume'].values
    
    # Get weekly data for signal generation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly RSI(7)
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/7, adjust=False, min_periods=7).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/7, adjust=False, min_periods=7).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily trend filter: price vs weekly EMA
    uptrend = close > ema_34_1w_aligned
    downtrend = close < ema_34_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 7  # Warmup for RSI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(rsi_aligned[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: weekly RSI < 30 (oversold) + weekly uptrend
            if rsi_aligned[i] < 30 and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly RSI > 70 (overbought) + weekly downtrend
            elif rsi_aligned[i] > 70 and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or trend change
            if rsi_aligned[i] > 50 or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or trend change
            if rsi_aligned[i] < 50 or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals