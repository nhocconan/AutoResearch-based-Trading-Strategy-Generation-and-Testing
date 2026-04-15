#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly RSI Filter
# Elder Ray measures bull power (high - EMA13) and bear power (low - EMA13).
# Long when bull power > 0 and bear power < 0 (bullish bias), short when bear power < 0 and bull power < 0 (bearish bias).
# Weekly RSI filter avoids counter-trend trades: only long when weekly RSI < 70, only short when weekly RSI > 30.
# Works in bull markets (buy strength) and bear markets (sell weakness). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA(13) for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load weekly data for RSI filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_weekly = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 6h timeframe
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(13, n):
        # Skip if weekly RSI data is not available
        if np.isnan(rsi_weekly_aligned[i]):
            continue
        
        # Long entry: bull power positive, bear power negative, and weekly RSI not overbought
        if (bull_power[i] > 0 and bear_power[i] < 0 and
            rsi_weekly_aligned[i] < 70 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bear power negative, bull power negative, and weekly RSI not oversold
        elif (bear_power[i] < 0 and bull_power[i] < 0 and
              rsi_weekly_aligned[i] > 30 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: power signals weaken or reverse
        elif position == 1 and (bull_power[i] <= 0 or bear_power[i] >= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power[i] >= 0 or bear_power[i] >= 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_WeeklyRSI_Filter"
timeframe = "6h"
leverage = 1.0