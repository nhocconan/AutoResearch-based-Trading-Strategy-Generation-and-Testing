#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly RSI extremes with daily trend filter
# Weekly RSI < 30 indicates oversold conditions on weekly timeframe, > 70 overbought
# Combined with daily EMA50 trend filter to avoid counter-trend trades
# Works in both bull/bear markets: buys oversold dips in uptrends, sells overbought rallies in downtrends
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyRSI30_70_DailyEMA50_TrendFilter_v1"
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
    
    # Calculate weekly RSI and daily EMA50 ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 14 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Daily EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly RSI < 30 (oversold) and price above daily EMA50 (uptrend)
            if rsi_1w_aligned[i] < 30 and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI > 70 (overbought) and price below daily EMA50 (downtrend)
            elif rsi_1w_aligned[i] > 70 and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly RSI > 50 (momentum fading) or price below EMA50 (trend change)
            if rsi_1w_aligned[i] > 50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly RSI < 50 (momentum fading) or price above EMA50 (trend change)
            if rsi_1w_aligned[i] < 50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals