# 6H_PivotBreakoutWithMomentumAndVolume
# Hypothesis: Daily pivot levels act as intraday support/resistance. Breakouts above R1 or below S1 with momentum (close > open) and volume confirmation (1.5x 20-period average) capture institutional flow. Weekly trend filter (price above/below weekly 20-period EMA) ensures alignment with higher timeframe trend, reducing whipsaw in chop. Target: 20-30 trades/year (80-120 total over 4 years) to stay within 6h limits. Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes via short entries).

name = "6H_PivotBreakoutWithMomentumAndVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Momentum: close > open (bullish) or close < open (bearish)
    momentum_long = close > open_price
    momentum_short = close < open_price
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # 1d data for daily pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from prior day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    rang = prev_high - prev_low
    R1 = pivot + rang
    S1 = pivot - rang
    
    # Align daily pivot levels to 6h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Weekly data for trend filter: 20-period EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_ema = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(20, n):  # Start after warmup for volume MA and weekly EMA
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(weekly_ema_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # LONG: Price breaks above R1 + bullish momentum + volume confirmation + price above weekly EMA (uptrend)
            if (close[i] > R1_aligned[i] and 
                momentum_long[i] and 
                volume_confirm[i] and 
                close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Price breaks below S1 + bearish momentum + volume confirmation + price below weekly EMA (downtrend)
            elif (close[i] < S1_aligned[i] and 
                  momentum_short[i] and 
                  volume_confirm[i] and 
                  close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters pivot range (between S1 and R1) OR closes below weekly EMA (trend change)
            if close[i] < R1_aligned[i] and close[i] > S1_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters pivot range (between S1 and R1) OR closes above weekly EMA (trend change)
            if close[i] < R1_aligned[i] and close[i] > S1_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals