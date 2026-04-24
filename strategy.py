#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend + RSI(14) extreme + Bollinger Band(20,2) squeeze regime filter.
- Long when KAMA rising (bullish trend) AND RSI < 30 (oversold) AND Bollinger Band width > 20th percentile (non-squeeze)
- Short when KAMA falling (bearish trend) AND RSI > 70 (overbought) AND Bollinger Band width > 20th percentile (non-squeeze)
- Uses 1d primary timeframe with 1w HTF for trend confirmation (KAMA on weekly close > weekly EMA50)
- Bollinger Band width regime filter avoids whipsaws in low-volatility environments
- Signal size: 0.25 discrete levels to minimize fee churn
- Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) - trend filter
    def kama(close, er_fast=2, er_slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
        # Correct calculation:
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.zeros_like(close)
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA on 1d
    kama_1d = kama(close, er_fast=2, er_slow=30)
    kama_rising = kama_1d > np.roll(kama_1d, 1)
    kama_falling = kama_1d < np.roll(kama_1d, 1)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Bollinger Band Width (20,2) - regime filter
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / ma_20
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 20) if len(x) == 50 else np.nan, raw=False
    ).values
    bb_width_above_threshold = bb_width > bb_width_percentile  # Non-squeeze regime
    
    # Get 1w data ONCE before loop for HTF trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_close_above_ema = df_1w['close'].values > ema_50_1w
    weekly_close_below_ema = df_1w['close'].values < ema_50_1w
    
    # Align weekly EMA50 to 1d timeframe (waits for completed weekly bar)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_above_ema.astype(float))
    weekly_bullish = weekly_trend_aligned > 0.5
    weekly_bearish = weekly_trend_aligned < 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need KAMA, RSI, BB width, and weekly alignment
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_rising[i]) or np.isnan(rsi[i]) or 
            np.isnan(bb_width_above_threshold[i]) or np.isnan(weekly_bullish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (bullish trend) AND RSI oversold AND non-squeeze regime AND weekly bullish
            if kama_rising[i] and rsi_oversold[i] and bb_width_above_threshold[i] and weekly_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (bearish trend) AND RSI overbought AND non-squeeze regime AND weekly bearish
            elif kama_falling[i] and rsi_overbought[i] and bb_width_above_threshold[i] and weekly_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA falling OR RSI overbought OR squeeze regime
            if (not kama_rising[i]) or rsi[i] > 70 or not bb_width_above_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA rising OR RSI oversold OR squeeze regime
            if (not kama_falling[i]) or rsi[i] < 30 or not bb_width_above_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_BBWidth_Regime_v1"
timeframe = "1d"
leverage = 1.0