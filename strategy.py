#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyKeltner_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly Keltner Channel (20-period, 2.0 ATR multiplier)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly
    tr1w = np.maximum(high_1w[1:] - low_1w[1:], 
                      np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                 np.abs(low_1w[1:] - close_1w[:-1])))
    tr1w = np.concatenate([[np.nan], tr1w])
    
    atr_1w = pd.Series(tr1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    middle_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = middle_1w + (2.0 * atr_1w)
    lower_keltner = middle_1w - (2.0 * atr_1w)
    
    # Align weekly Keltner bands to 12h
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    middle_keltner_aligned = align_htf_to_ltf(prices, df_1w, middle_1w)
    
    # Daily RSI(14) for mean reversion signals
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily close for trend filter
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        keltner_upper = upper_keltner_aligned[i]
        keltner_lower = lower_keltner_aligned[i]
        keltner_middle = middle_keltner_aligned[i]
        rsi = rsi_1d_aligned[i]
        daily_close = close_1d_aligned[i]
        
        if position == 0:
            # Enter long: price touches lower Keltner + RSI oversold + price above daily close (bullish bias)
            if (price <= keltner_lower and rsi < 30 and price > daily_close):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper Keltner + RSI overbought + price below daily close (bearish bias)
            elif (price >= keltner_upper and rsi > 70 and price < daily_close):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches middle Keltner OR RSI overbought
            if (price >= keltner_middle or rsi > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches middle Keltner OR RSI oversold
            if (price <= keltner_middle or rsi < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Uses weekly Keltner channel touches with daily RSI extremes and daily trend filter for mean reversion.
# - Enters long when price touches lower weekly Keltner band with RSI < 30 and price above daily close
# - Enters short when price touches upper weekly Keltner band with RSI > 70 and price below daily close
# - Exits when price returns to weekly Keltner middle OR RSI reaches opposite extreme
# - Weekly Keltner provides dynamic support/resistance based on volatility
# - Daily RSI ensures entry at extreme momentum conditions
# - Daily close filter ensures trading with higher timeframe trend direction
# - Mean reversion strategy works in both bull and bear markets by fading extremes
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return
# - Works on BTC and ETH as primary targets (not relying on SOL's unique behavior)