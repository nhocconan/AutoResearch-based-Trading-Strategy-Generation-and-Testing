#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily KAMA + RSI + Chop regime for 1d timeframe
# Long when KAMA rising + RSI < 40 + Chop > 61.8 (range) for mean reversion
# Short when KAMA falling + RSI > 60 + Chop > 61.8 (range) for mean reversion
# Exit when Chop < 38.2 (trending) or RSI crosses 50
# Uses weekly trend filter: only trade long when price > weekly EMA50, short when price < weekly EMA50
# Designed for low trade frequency (~10-25/year) to minimize fee drag and work in range-bound markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate 50-period EMA on weekly close for trend filter
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Load daily data for KAMA, RSI, Chop
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |close - close_10| / sum(|close - close-1|) over 10 periods
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    direction = np.abs(np.diff(close_daily, n=10, prepend=close_daily[:10]))
    er = np.where(volatility != 0, direction / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # Calculate RSI(14)
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Calculate Chop Chopiness Index(14)
    # Chop = 100 * log10(sum(ATR1) / (n * log10(highest_high - lowest_low)))
    tr1 = np.maximum(high_daily - low_daily, 
                     np.maximum(np.abs(high_daily - np.roll(close_daily, 1)), 
                                np.abs(low_daily - np.roll(close_daily, 1))))
    tr1[0] = high_daily[0] - low_daily[0]
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_daily).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_daily).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl > 0, 100 * np.log10(atr1 / 14) / np.log10(range_hl), 50)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(ema_50_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_weekly = ema_50_weekly_aligned[i]
        
        # Regime filter: only trade in ranging markets (Chop > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Long conditions: KAMA rising + RSI oversold + in range + weekly uptrend filter
            if (price > kama_val and  # price above KAMA (KAMA rising proxy)
                rsi_val < 40 and 
                in_range and 
                price > ema_weekly):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling + RSI overbought + in range + weekly downtrend filter
            elif (price < kama_val and  # price below KAMA (KAMA falling proxy)
                  rsi_val > 60 and 
                  in_range and 
                  price < ema_weekly):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Chop < 38.2 (trending) or RSI crosses 50
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Chop < 38.2 (trending) or RSI >= 50
                if chop_val < 38.2 or rsi_val >= 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Chop < 38.2 (trending) or RSI <= 50
                if chop_val < 38.2 or rsi_val <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_WeeklyEMA50_Filter"
timeframe = "1d"
leverage = 1.0