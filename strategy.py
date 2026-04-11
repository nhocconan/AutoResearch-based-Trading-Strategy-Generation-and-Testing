#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day/1-week KAMA trend + RSI + chop filter.
# Uses weekly KAMA for trend direction, daily RSI for mean reversion entry,
# and weekly chop filter to avoid trending markets. Designed for 15-30 trades/year.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate.

name = "12h_1w1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend direction
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=1)  # 10-period volatility
    er = np.zeros_like(close_1w)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    # Weekly trend: price above KAMA = bullish, below = bearish
    weekly_trend_bull = close_1w > kama
    weekly_trend_bear = close_1w < kama
    
    # Align weekly trend to 12h
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    
    # Calculate daily RSI for mean reversion
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # RSI oversold/overbought
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Align daily RSI to 12h
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_1d, rsi_oversold)
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_1d, rsi_overbought)
    
    # Calculate weekly chop filter (Ehler's Chop Index)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # True range
    tr1 = np.abs(np.diff(high_1w))
    tr2 = np.abs(np.diff(low_1w))
    tr3 = np.abs(np.diff(close_1w))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Add first element
    tr = np.concatenate([[np.abs(high_1w[0] - low_1w[0])], tr])
    # Sum of true ranges over 14 periods
    atr_sum = np.zeros_like(close_1w)
    for i in range(14, len(tr)):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    # Absolute price change over 14 periods
    price_change = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        price_change[i] = np.abs(close_1w[i] - close_1w[i-14])
    # Chop index
    chop = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        if atr_sum[i] > 0:
            chop[i] = 100 * np.log10(price_change[i] / atr_sum[i]) / np.log10(14)
        else:
            chop[i] = 50
    # Chop > 61.8 = ranging market (good for mean reversion)
    chop_range = chop > 61.8
    
    # Align chop filter to 12h
    chop_range_aligned = align_htf_to_ltf(prices, df_1w, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_oversold_aligned[i]) or np.isnan(rsi_overbought_aligned[i]) or
            np.isnan(chop_range_aligned[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop filter: only trade in ranging markets
        in_range = chop_range_aligned[i]
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # Mean reversion entries in ranging markets
        long_entry = (rsi_oversold_aligned[i] and in_range and is_bullish_week)
        short_entry = (rsi_overbought_aligned[i] and in_range and is_bearish_week)
        
        # Exit when RSI returns to neutral or trend changes
        exit_long = (position == 1 and 
                    (rsi[i] >= 50 or not is_bullish_week))
        exit_short = (position == -1 and 
                     (rsi[i] <= 50 or not is_bearish_week))
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals