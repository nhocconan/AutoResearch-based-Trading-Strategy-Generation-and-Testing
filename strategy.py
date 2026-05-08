#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = (close > kama).astype(float)
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chopiness Index (14) - regime filter
    atr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(trend_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, chop < 61.8 (trending), weekly uptrend
            long_cond = (kama_dir[i] > 0.5 and rsi[i] > 50 and chop[i] < 61.8 and trend_1w_aligned[i] > 0.5)
            # Short: KAMA down, RSI < 50, chop < 61.8 (trending), weekly downtrend
            short_cond = (kama_dir[i] < 0.5 and rsi[i] < 50 and chop[i] < 61.8 and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down or RSI < 40
            if kama_dir[i] < 0.5 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up or RSI > 60
            if kama_dir[i] > 0.5 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend + RSI momentum + Chop regime filter + weekly trend filter.
# KAMA adapts to market conditions, reducing whipsaws. RSI >50/<50 filters momentum.
# Chop < 61.8 ensures we only trade in trending markets, avoiding chop.
# Weekly EMA34 ensures alignment with long-term trend.
# Works in bull markets (trend following) and bear markets (avoids false signals via chop filter).
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.