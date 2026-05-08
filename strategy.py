#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_Trend_v1"
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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly trend: weekly close > weekly open = uptrend
    prev_week_open = np.roll(df_1w['open'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_open[0] = df_1w['open'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    weekly_trend = prev_week_close > prev_week_open
    
    # Align weekly trend to daily timeframe
    weekly_trend_1d = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # KAMA calculation on daily closes
    close_series = pd.Series(close)
    change = np.abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_1d = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) calculation
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if atr[i] > 0 and highest_high[i] > lowest_low[i]:
            sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum()[i]
            chop[i] = 100 * np.log10(sum_tr / (atr[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(weekly_trend_1d[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above KAMA, RSI > 50, chop < 50 (trending), volume spike, weekly uptrend
            long_cond = (close[i] > kama_1d[i] and rsi[i] > 50 and chop[i] < 50 and 
                         vol_spike[i] and weekly_trend_1d[i] > 0.5)
            
            # Short entry: price below KAMA, RSI < 50, chop < 50 (trending), volume spike, weekly downtrend
            short_cond = (close[i] < kama_1d[i] and rsi[i] < 50 and chop[i] < 50 and 
                          vol_spike[i] and weekly_trend_1d[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or chop > 60 (choppy market)
            if close[i] < kama_1d[i] or chop[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or chop > 60 (choppy market)
            if close[i] > kama_1d[i] or chop[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend + RSI momentum + chop regime filter with volume spike confirmation on 1d timeframe.
# Enters long when price > KAMA, RSI > 50, chop < 50 (trending), volume spike, and weekly uptrend.
# Enters short when price < KAMA, RSI < 50, chop < 50 (trending), volume spike, and weekly downtrend.
# Exits when price crosses KAMA or market becomes choppy (chop > 60).
# Weekly trend filter ensures alignment with higher timeframe direction.
# Uses discrete sizing (0.25) to minimize churn. Targets 10-20 trades/year on 1d timeframe.
# Works in bull markets (trend following) and bear markets (counter-trend in choppy conditions avoided).