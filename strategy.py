#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily strategy using weekly ATR breakout with volume confirmation and 1-week RSI trend filter.
# Weekly ATR breakout captures major momentum moves while filtering noise.
# Volume > 2.0x 20-day average confirms institutional participation.
# Weekly RSI > 50 for long, < 50 for short ensures alignment with weekly momentum.
# Designed for ~10-20 trades/year by requiring significant ATR-based breakouts.
# Works in bull/bear: buys breakouts above weekly ATR resistance, sells breakdowns below weekly ATR support.
# Exit when price reverts to weekly midpoint or RSI reverses.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR and RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly midpoint (average of high and low)
    midpoint_1w = (high_1w + low_1w) / 2.0
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align weekly indicators to daily
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    # Volume filter: volume > 2.0 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly ATR (14), midpoint (0), RSI (14), volume MA (20)
    start_idx = max(15, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Weekly trend filters
        weekly_bullish = rsi_14_aligned[i] > 50
        weekly_bearish = rsi_14_aligned[i] < 50
        
        # Breakout levels
        upper_breakout = midpoint_aligned[i] + atr_14_aligned[i]
        lower_breakout = midpoint_aligned[i] - atr_14_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly midpoint + ATR with volume and weekly bullish
            if price > upper_breakout and vol_filter and weekly_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly midpoint - ATR with volume and weekly bearish
            elif price < lower_breakout and vol_filter and weekly_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below weekly midpoint or weekly RSI turns bearish
            if price < midpoint_aligned[i] or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above weekly midpoint or weekly RSI turns bullish
            if price > midpoint_aligned[i] or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyATR_Breakout_Volume_RSI"
timeframe = "1d"
leverage = 1.0