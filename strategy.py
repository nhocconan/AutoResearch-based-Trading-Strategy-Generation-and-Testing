#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_Regime
Hypothesis: Daily timeframe strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index regime filter to avoid sideways markets.
KAMA adapts to market noise, reducing whipsaws in choppy conditions. RSI filters extreme entries.
Chop regime ensures we only trade when market is trending (CHOP < 38.2) or mean-reverting (CHOP > 61.8)
depending on signal type. Designed for low frequency (target 15-25 trades/year) to minimize fee drag.
Works in both bull and bear markets via adaptive trend and regime filters.
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on daily close
    # KAMA parameters: ER fast=2, slow=30, lookback=10
    fast = 2
    slow = 30
    lookback = 10
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first lookback periods
    er = np.zeros_like(close)
    er[lookback:] = change / np.where(volatility[lookback:] == 0, 1, volatility[lookback:])
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_aligned = kama  # KAMA is already on 1d timeframe
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Calculate Choppiness Index (CHOP) on daily data
    chop_period = 14
    atr_chop = np.zeros_like(close)
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    
    # True Range
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # ATR and max/min over period
    for i in range(chop_period, len(close)):
        atr_chop[i] = np.sum(tr[i-chop_period+1:i+1])
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    # Chop formula: 100 * log10(ATR_sum / (max_high - min_low)) / log10(period)
    chop = np.zeros_like(close)
    for i in range(chop_period, len(close)):
        if max_high[i] - min_low[i] > 0:
            chop[i] = 100 * np.log10(atr_chop[i] / (max_high[i] - min_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50  # Neutral
    
    chop[:chop_period] = 50  # Neutral before enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations
    start_idx = max(lookback, 14, chop_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        # Determine trend: price above/below KAMA
        bullish = close_val > kama_val
        bearish = close_val < kama_val
        
        # Determine 1w HTF trend
        htf_bullish = close_val > ema_50_1w_val  # Approximate: using current price vs 1w EMA
        htf_bearish = close_val < ema_50_1w_val
        
        # Regime filters
        # Chop < 38.2 = trending (good for trend following)
        # Chop > 61.8 = ranging (good for mean reversion)
        trending_regime = chop_val < 38.2
        ranging_regime = chop_val > 61.8
        
        # RSI filters: avoid extremes
        rsi_not_overbought = rsi_val < 70
        rsi_not_oversold = rsi_val > 30
        
        # Entry logic:
        # In trending regime: follow KAMA trend with RSI momentum
        # In ranging regime: mean revert at RSI extremes
        long_entry = False
        short_entry = False
        
        if trending_regime:
            # Trend following: long when bullish alignment, short when bearish
            long_entry = bullish and htf_bullish and rsi_not_overbought and rsi_val > 50
            short_entry = bearish and htf_bearish and rsi_not_oversold and rsi_val < 50
        elif ranging_regime:
            # Mean reversion: long at RSI oversold, short at RSI overbought
            long_entry = rsi_val < 30 and not htf_bearish  # Avoid fighting strong HTF downtrend
            short_entry = rsi_val > 70 and not htf_bullish  # Avoid fighting strong HTF uptrend
        
        # Exit conditions: opposite signal or regime change to choppy
        choppy_regime = chop_val >= 38.2 and chop_val <= 61.8
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (not bullish or not htf_bullish or rsi_val >= 70 or choppy_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not bearish or not htf_bearish or rsi_val <= 30 or choppy_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_Regime"
timeframe = "1d"
leverage = 1.0