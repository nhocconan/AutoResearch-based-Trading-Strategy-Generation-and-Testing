#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band Squeeze + RSI Mean Reversion with Weekly Trend Filter
# In low volatility (Bollinger Band width < 20th percentile), price tends to revert to mean.
# Long when RSI < 30 and price touches lower BB, short when RSI > 70 and price touches upper BB.
# Weekly trend filter: only trade in direction of weekly trend (price above/below weekly EMA50).
# Works in both bull and bear markets by capturing mean reversion within the prevailing trend.
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Bollinger Bands (20, 2) on 1d
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (std * bb_std)
    lower_bb = sma - (std * bb_std)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_bb - lower_bb) / sma
    # Squeeze when BB width is below 20th percentile of its 50-period lookback
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).rank(pct=True).values
    squeeze = bb_width_percentile < 0.2  # True when in squeeze (low volatility)
    
    # RSI (14) on 1d
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Weekly trend: price above EMA50 = uptrend, below = downtrend
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align 1d indicators to lower timeframe (1d is our base timeframe, so no alignment needed for 1d data)
    # But we need to align weekly data to 1d timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0  # 1 for long, -1 for short, 0 for flat
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            continue
        
        # Long entry: squeeze + RSI oversold + price at lower BB + weekly uptrend
        if (squeeze_aligned[i] and
            rsi_aligned[i] < 30 and
            close[i] <= lower_bb_aligned[i] and
            weekly_uptrend_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: squeeze + RSI overbought + price at upper BB + weekly downtrend
        elif (squeeze_aligned[i] and
              rsi_aligned[i] > 70 and
              close[i] >= upper_bb_aligned[i] and
              weekly_downtrend_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral range (40-60) or squeeze ends
        elif position == 1 and (rsi_aligned[i] > 40 or not squeeze_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] < 60 or not squeeze_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Squeeze_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0