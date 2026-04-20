#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Heikin-Ashi smoothed EMA crossover with 12h RSI filter
# Uses Heikin-Ashi candles to reduce noise and identify true trends
# EMA(9) crossing EMA(21) on HA close provides entry signals
# 12h RSI(14) acts as trend strength filter (RSI>50 for long, RSI<50 for short)
# Designed to work in both bull and bear markets by following the trend
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE for RSI filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI(14)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate Heikin-Ashi candles
    ha_close = (prices['open'] + prices['high'] + prices['low'] + prices['close']) / 4
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (prices['open'].iloc[0] + prices['close'].iloc[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(prices['high'], np.maximum(ha_open, ha_close))
    ha_low = np.minimum(prices['low'], np.minimum(ha_open, ha_close))
    
    # Calculate EMA(9) and EMA(21) on HA close
    ha_close_series = pd.Series(ha_close.values)
    ema9 = ha_close_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = ha_close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6h ATR for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if NaN in indicators
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(rsi_12h_aligned[i]) or np.isnan(atr_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        price = ha_close[i]
        rsi = rsi_12h_aligned[i]
        
        if position == 0:
            # Long: EMA9 crosses above EMA21 and RSI > 50 (bullish momentum)
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and rsi > 50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: EMA9 crosses below EMA21 and RSI < 50 (bearish momentum)
            elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and rsi < 50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: EMA9 crosses below EMA21 or stop loss (2x ATR)
            if ema9[i] < ema21[i] or price <= entry_price - 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA9 crosses above EMA21 or stop loss (2x ATR)
            if ema9[i] > ema21[i] or price >= entry_price + 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HA_EMA9_21_RSI12h_Filter"
timeframe = "6h"
leverage = 1.0