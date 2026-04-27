#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter
Hypothesis: KAMA adapts to market noise, RSI identifies extremes, and Choppiness Index filters regimes.
In chop (high CHOP), mean-revert at RSI extremes; in trend (low CHOP), follow KAMA direction.
Works in bull/bear via adaptive filtering. Target: 10-20 trades/year on 1d.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # KAMA(14, 2, 30) - adaptive moving average
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(change))  # not correct, need proper volatility
    # Proper ER calculation
    price_change = np.abs(np.subtract(close[14:], close[:-14]))  # |close_t - close_{t-14}|
    volatility_sum = np.array([
        np.sum(np.abs(np.diff(close[i:i+14]))) if i+14 <= len(close) else np.nan
        for i in range(len(close))
    ])
    # Fix volatility calculation
    volatility = np.array([
        np.sum(np.abs(np.diff(close[max(0, i-13):i+1]))) if i >= 13 else np.nan
        for i in range(len(close))
    ])
    er = np.where(volatility > 0, price_change / volatility, 0)
    # Fill beginning with NaN
    er = np.concatenate([np.full(13, np.nan), er[13:]])
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.array([
        np.max([
            high[i] - low[i],
            np.abs(high[i] - close[i-1]) if i > 0 else 0,
            np.abs(low[i] - close[i-1]) if i > 0 else 0
        ]) for i in range(len(close))
    ])
    atr_sum = np.array([
        np.sum(atr[max(0, i-13):i+1]) if i >= 13 else np.nan
        for i in range(len(close))
    ])
    highest_high = np.array([
        np.max(high[max(0, i-13):i+1]) if i >= 13 else np.nan
        for i in range(len(close))
    ])
    lowest_low = np.array([
        np.min(low[max(0, i-13):i+1]) if i >= 13 else np.nan
        for i in range(len(close))
    ])
    chop = np.where(
        (highest_high - lowest_low) > 0,
        100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14),
        50
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        weekly_trend = ema34_1w_aligned[i]
        
        if position == 0:
            # In chop (chop > 61.8): mean reversion at RSI extremes
            if chop_val > 61.8:
                if rsi_val < 30 and close[i] > kama_val:  # oversold + price above KAMA
                    signals[i] = size
                    position = 1
                elif rsi_val > 70 and close[i] < kama_val:  # overbought + price below KAMA
                    signals[i] = -size
                    position = -1
            # In trend (chop < 38.2): follow weekly trend
            elif chop_val < 38.2:
                if close[i] > kama_val and close[i] > weekly_trend:  # above KAMA and weekly trend
                    signals[i] = size
                    position = 1
                elif close[i] < kama_val and close[i] < weekly_trend:  # below KAMA and weekly trend
                    signals[i] = -size
                    position = -1
            # In transition zone: no action
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi_val > 70 or close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi_val < 30 or close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0