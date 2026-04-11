#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Hypothesis: 1d KAMA direction + RSI(14) + Choppiness index filter
# - KAMA adapts to market noise, smooth in trend, responsive in range
# - Long when KAMA rising + RSI > 50 + CHOP < 61.8 (trending regime)
# - Short when KAMA falling + RSI < 50 + CHOP < 61.8 (trending regime)
# - Avoids choppy markets (CHOP > 61.8) where trend signals fail
# - Uses 1-week trend filter to avoid counter-trend trades
# - Discrete position sizing ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits for 1d
# - Works in bull markets (trend continuation) and bear markets (trend continuation with 1w filter)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data (same as primary) for indicators
    df_1d = prices  # primary timeframe is 1d
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute KAMA (1d)
    # Efficiency ratio
    change = np.abs(np.diff(close, n=1))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    price_change = np.abs(close - np.roll(close, 1))
    price_change[0] = 0
    volatility_sum = np.convolve(np.abs(np.diff(close)), np.ones(10), 'same')  # temporary
    
    # Recalculate properly
    change_t = np.abs(np.diff(close, n=1))
    volatility_t = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_t[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility_t[i] > 0:
            er[i] = change_t[i] / volatility_t[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rs[13:] = avg_gain[13:] / np.where(avg_loss[13:] == 0, 1, avg_loss[13:])
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute Choppiness Index (14)
    atr_t = np.zeros_like(close)
    for i in range(1, len(close)):
        atr_t[i] = max(
            high[i] - low[i],
            np.abs(high[i] - close[i-1]),
            np.abs(low[i] - close[i-1])
        )
    
    # True range sum over 14 periods
    tr_sum = np.zeros_like(close)
    for i in range(13, len(close)):
        tr_sum[i] = np.sum(atr_t[i-12:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(len(close)):
        if i < 13:
            max_high[i] = np.max(high[:i+1])
            min_low[i] = np.min(low[:i+1])
        else:
            max_high[i] = np.max(high[i-12:i+1])
            min_low[i] = np.min(low[i-12:i+1])
    
    # Chop calculation
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(30, n):  # Start after 30-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        kama_prev = kama[i-1] if i > 0 else kama[i]
        
        # KAMA direction
        kama_rising = kama[i] > kama_prev
        kama_falling = kama[i] < kama_prev
        
        # RSI condition
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        
        # Choppiness regime: trending when CHOP < 61.8
        trending_regime = chop[i] < 61.8
        
        # 1-week trend filter
        price_above_1w_ema50 = price_close > ema50_1w_aligned[i]
        price_below_1w_ema50 = price_close < ema50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA rising + RSI > 50 + trending regime + price above 1w EMA50
        if kama_rising and rsi_above_50 and trending_regime and price_above_1w_ema50:
            enter_long = True
        
        # Short: KAMA falling + RSI < 50 + trending regime + price below 1w EMA50
        if kama_falling and rsi_below_50 and trending_regime and price_below_1w_ema50:
            enter_short = True
        
        # Exit conditions: opposite KAMA direction or choppy regime
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if KAMA falling OR choppy regime
            exit_long = kama_falling or (chop[i] >= 61.8)
        elif position == -1:
            # Exit short if KAMA rising OR choppy regime
            exit_short = kama_rising or (chop[i] >= 61.8)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals