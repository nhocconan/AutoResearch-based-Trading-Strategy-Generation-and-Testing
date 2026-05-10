#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_ChopFilter_v1
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI for momentum filter, and Choppiness Index for regime filtering on daily timeframe. Designed to capture trends while avoiding whipsaws in choppy markets. Targets 15-25 trades/year to minimize fee drag. Works in both bull and bear markets by adapting trend sensitivity and using mean-reversion in ranging conditions.

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (2-period ER, 30-period smoothing)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(change) > 0 else np.array([])
    # Handle array dimensions
    if len(change) > 0 and len(volatility) > 0:
        er = np.where(volatility != 0, change / volatility, 0)
    else:
        er = np.zeros_like(change)
    # Pad er to match close length
    er_padded = np.concatenate([np.full(10, np.nan), er])
    sc = (er_padded * 0.0645) ** 2  # (2/(2+1))^2 = 0.0645, (30/(30+1))^2 ≈ 0.946
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi_padded = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index (14-period)
    atr1 = high - low
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr1[0] = 0
    atr2[0] = 0
    atr3[0] = 0
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    # Pad CHOP to match length
    chop_padded = np.concatenate([np.full(13, np.nan), chop])
    
    # Get weekly EMA for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi_padded[i]) or np.isnan(chop_padded[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Momentum filter
        rsi_middle = 40 < rsi_padded[i] < 60
        rsi_overbought = rsi_padded[i] > 60
        rsi_oversold = rsi_padded[i] < 40
        
        # Regime filter: Chop > 61.8 = range, Chop < 38.2 = trend
        chop_high = chop_padded[i] > 61.8  # Choppy/ranging
        chop_low = chop_padded[i] < 38.2   # Trending
        
        if position == 0:
            # Long: KAMA uptrend + weekly uptrend + RSI not overbought + trending OR (choppy + RSI oversold)
            if (price_above_kama and weekly_uptrend and not rsi_overbought and (chop_low or (chop_high and rsi_oversold))):
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + weekly downtrend + RSI not oversold + trending OR (choppy + RSI overbought)
            elif (price_below_kama and weekly_downtrend and not rsi_oversold and (chop_low or (chop_high and rsi_overbought))):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA downtrend OR weekly downtrend OR RSI overbought in chop
            if price_below_kama or not weekly_uptrend or (chop_high and rsi_overbought):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA uptrend OR weekly uptrend OR RSI oversold in chop
            if price_above_kama or not weekly_downtrend or (chop_high and rsi_oversold):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals