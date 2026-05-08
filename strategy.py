#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Hull Moving Average (HMA) crossover with RSI filter and weekly trend confirmation
# Long when HMA(21) crosses above HMA(55) and RSI < 55 in weekly uptrend
# Short when HMA(21) crosses below HMA(55) and RSI > 45 in weekly downtrend
# HMA reduces lag while maintaining smoothness, RSI filters overextension, weekly trend ensures alignment
# Targets 30-100 trades over 4 years to minimize fee drag while capturing major trends

name = "1d_HMA_Crossover_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter (more responsive than SMA)
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # HMA(21) - Hull Moving Average
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, 'valid') / weights.sum()
    
    def hma(data, window):
        half = window // 2
        sqrt = int(np.sqrt(window))
        wma_half = wma(data, half)
        wma_full = wma(data, window)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt)
    
    # Pad arrays to match original length
    hma_21_raw = hma(close, 21)
    hma_55_raw = hma(close, 55)
    hma_21 = np.full_like(close, np.nan)
    hma_55 = np.full_like(close, np.nan)
    hma_21[20:] = hma_21_raw  # HMA(21) needs 21 periods
    hma_55[54:] = hma_55_raw  # HMA(55) needs 55 periods
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # warmup for HMA(55)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(hma_21[i]) or np.isnan(hma_55[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hma_21_val = hma_21[i]
        hma_55_val = hma_55[i]
        rsi_val = rsi[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: HMA(21) > HMA(55) and RSI < 55 in weekly uptrend
            if hma_21_val > hma_55_val and rsi_val < 55 and ema34_1w_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: HMA(21) < HMA(55) and RSI > 45 in weekly downtrend
            elif hma_21_val < hma_55_val and rsi_val > 45 and ema34_1w_val < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: HMA(21) < HMA(55) or RSI > 70 (overbought) or weekly trend down
            if hma_21_val < hma_55_val or rsi_val > 70 or ema34_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: HMA(21) > HMA(55) or RSI < 30 (oversold) or weekly trend up
            if hma_21_val > hma_55_val or rsi_val < 30 or ema34_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals