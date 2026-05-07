# 1d_KAMA_RSI_ChopFilter_v2
# Hypothesis: On 1d timeframe, use KAMA for adaptive trend direction, RSI for momentum exhaustion, and Choppiness Index for regime filtering.
# In choppy markets (high CHOP), mean-revert at RSI extremes with the trend; in trending markets (low CHOP), follow breakouts.
# This reduces whipsaws in ranging markets and captures momentum in trends, suitable for both bull and bear markets.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.

name = "1d_KAMA_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0

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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA (adaptive moving average) for trend
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation over 10 periods
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            sum_abs_diff = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = price_change / (sum_abs_diff + 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[i-13:i+1])
            avg_loss[i] = np.mean(loss[i-13:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    # Sum of true range over 14 periods
    sum_tr14 = np.zeros(n)
    for i in range(14, n):
        sum_tr14[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    max_high14 = np.zeros(n)
    min_low14 = np.zeros(n)
    for i in range(14, n):
        max_high14[i] = np.max(high[i-13:i+1])
        min_low14[i] = np.min(low[i-13:i+1])
    chop = np.zeros(n)
    for i in range(14, n):
        if sum_tr14[i] > 0 and (max_high14[i] - min_low14[i]) > 0:
            chop[i] = 100 * np.log10(sum_tr14[i] / (max_high14[i] - min_low14[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral if undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 10)  # ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        trend_up = close > ema_50_1w_aligned[i]
        trend_down = close < ema_50_1w_aligned[i]
        
        # Chop regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (follow momentum)
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Long conditions
            long_signal = False
            if is_ranging:
                # In ranging: mean revert at RSI oversold with upward momentum
                if rsi[i] < 30 and close[i] > kama[i]:
                    long_signal = True
            else:  # trending or neutral chop
                # In trending: follow trend with pullback to KAMA
                if trend_up[i] and close[i] > kama[i] and rsi[i] > 50:
                    long_signal = True
            
            # Short conditions
            short_signal = False
            if is_ranging:
                # In ranging: mean revert at RSI overbought with downward momentum
                if rsi[i] > 70 and close[i] < kama[i]:
                    short_signal = True
            else:  # trending or neutral chop
                # In trending: follow trend with pullback to KAMA
                if trend_down[i] and close[i] < kama[i] and rsi[i] < 50:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend change or price below KAMA
            if rsi[i] > 70 or not trend_up[i] or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or trend change or price above KAMA
            if rsi[i] < 30 or not trend_down[i] or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals