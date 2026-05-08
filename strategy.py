#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_ChopFilter_1wTrend"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 trend filter (using weekly close)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # KAMA calculation on daily prices
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close(t) - close(t-1)| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # indices 1-13
    avg_loss[13] = np.mean(loss[1:14])
    # Subsequent averages
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    atr_sum = np.convolve(tr, np.ones(14), 'full')[:n]  # simple sum
    atr_sum[:13] = np.nan  # not enough data
    # Highest high and lowest low over 14 periods
    highest_high = np.concatenate([[np.nan]*13, np.max(np.lib.stride_tricks.sliding_window_view(high, 14), axis=1)])
    lowest_low = np.concatenate([[np.nan]*13, np.min(np.lib.stride_tricks.sliding_window_view(low, 14), axis=1)])
    # Chop calculation
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, chop < 61.8 (trending), weekly uptrend
            long_cond = (close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8 and trend_1w_aligned[i] > 0.5)
            # Short: price < KAMA, RSI < 50, chop < 61.8 (trending), weekly downtrend
            short_cond = (close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8 and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA OR RSI < 40 OR chop > 61.8 (ranging) OR weekly trend change
            if (close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8 or trend_1w_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA OR RSI > 60 OR chop > 61.8 (ranging) OR weekly trend change
            if (close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8 or trend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily KAMA trend filter with RSI momentum and Choppiness Index regime filter,
# aligned with weekly EMA34 trend. Enters long when price > KAMA, RSI > 50, chop < 61.8 (trending),
# and weekly uptrend. Enters short when price < KAMA, RSI < 50, chop < 61.8 (trending),
# and weekly downtrend. Exits when trend fails (price crosses KAMA, RSI extremes, chop > 61.8 ranging,
# or weekly trend change). Uses discrete sizing (0.25) to minimize churn. Targets 15-25 trades/year
# on daily timeframe to avoid overtrading. Works in bull markets (trend following) and bear markets
# (avoids ranging markets via chop filter, captures trends when they occur). KAMA adapts to market
# noise, reducing false signals in choppy conditions. Weekly trend filter ensures alignment with
# higher timeframe momentum, improving reliability across market regimes.