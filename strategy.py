#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use KAMA (adaptive trend) for direction, RSI for momentum, and Choppiness Index to filter ranging markets.
Long when KAMA > price, RSI > 50, and CHOP > 61.8 (ranging) for mean reversion to upside.
Short when KAMA < price, RSI < 50, and CHOP > 61.8 (ranging) for mean reversion to downside.
In trending markets (CHOP < 38.2), follow KAMA direction only.
Uses weekly trend filter: only take longs when weekly close > weekly EMA20, shorts when weekly close < weekly EMA20.
Designed for low turnover (~10-25 trades/year) with edge in both bull (trend follow) and bear (mean revert in ranges).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    if len(close_1w) >= 20:
        ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
        weekly_trend_up = close_1w > ema_20_1w
        weekly_trend_down = close_1w < ema_20_1w
    else:
        ema_20_1w = np.full_like(close_1w, np.nan)
        weekly_trend_up = np.zeros_like(close_1w, dtype=bool)
        weekly_trend_down = np.zeros_like(close_1w, dtype=bool)
    
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # KAMA (adaptive trend) - using ER = 10, fast=2, slow=30
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([])
        # Pad volatility to match change length
        if len(volatility) < len(change):
            volatility = np.concatenate([volatility, np.full(len(change) - len(volatility), np.nan)])
        elif len(volatility) > len(change):
            volatility = volatility[:len(change)]
        
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        kama_out = np.full_like(close, np.nan)
        kama_out[er_period] = close[er_period]  # seed
        
        for i in range(er_period + 1, len(close)):
            if not np.isnan(sc[i-er_period]):
                kama_out[i] = kama_out[i-1] + sc[i-er_period] * (close[i-1] - kama_out[i-1])
            else:
                kama_out[i] = kama_out[i-1]
        return kama_out
    
    kama_val = kama(close, 10, 2, 30)
    
    # RSI(14)
    def rsi(close, period=14):
        if len(close) < period + 1:
            return np.full_like(close, np.nan)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_val = rsi(close, 14)
    
    # Choppiness Index
    def chop(high, low, close, period=14):
        if len(close) < period:
            return np.full_like(close, np.nan)
        atr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
        atr[0] = high[0] - low[0]  # first ATR
        
        # True range sum over period
        tr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            tr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_h = np.zeros_like(close)
        min_l = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_h[i] = np.max(high[i-period+1:i+1])
            min_l[i] = np.min(low[i-period+1:i+1])
        
        chop_val = np.where(
            (max_h - min_l) != 0,
            100 * np.log10(tr_sum / (max_h - min_l)) / np.log10(period),
            50
        )
        return chop_val
    
    chop_val = chop(high, low, close, 14)
    
    # Align all indicators to daily (they're already daily, but for consistency)
    kama_aligned = kama_val  # already aligned to prices
    rsi_aligned = rsi_val
    chop_aligned = chop_val
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = max(50, 20)  # KAMA needs ~50, weekly EMA20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i])):
            continue
        
        # Conditions
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        chop_high = chop_aligned[i] > 61.8  # ranging market
        chop_low = chop_aligned[i] < 38.2   # trending market
        weekly_up = weekly_trend_up_aligned[i] > 0.5
        weekly_down = weekly_trend_down_aligned[i] > 0.5
        
        # Exit conditions (reverse position)
        if close[i] > kama_aligned[i] * 1.05:  # 5% above KAMA - take profit
            signals[i] = 0.0
            continue
        if close[i] < kama_aligned[i] * 0.95:  # 5% below KAMA - take profit
            signals[i] = 0.0
            continue
            
        # Entry logic
        if chop_high:  # ranging market - mean reversion
            if price_below_kama and rsi_below_50 and weekly_up:
                signals[i] = 0.25  # long
            elif price_above_kama and rsi_above_50 and weekly_down:
                signals[i] = -0.25  # short
        elif chop_low:  # trending market - follow trend
            if price_above_kama and rsi_above_50 and weekly_up:
                signals[i] = 0.25  # long
            elif price_below_kama and rsi_below_50 and weekly_down:
                signals[i] = -0.25  # short
        # In neutral chop (38.2-61.8), no new entries but may hold
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0