#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly RSI mean reversion with daily price action confirmation.
# In bear markets, weekly RSI >70 signals exhaustion for short entries; RSI <30 signals bounce for longs.
# Daily inside day pattern (lower high and higher low) confirms rejection of extreme levels.
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Target: 50-150 total trades over 4 years (12-38/year).

name = "exp_13564_1w_rsi_meanrev_1d_inside_day_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_RSI_PERIOD = 14
WEEKLY_RSI_OVERBOUGHT = 70
WEEKLY_RSI_OVERSOLD = 30
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI for mean reversion signals
    close_1w = df_1w['close'].values
    rsi_1w = calculate_rsi(close_1w, WEEKLY_RSI_PERIOD)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_RSI_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly RSI not available
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Daily inside day pattern: lower high AND higher low
        inside_day = (high[i] < high[i-1]) and (low[i] > low[i-1])
        
        # Mean reversion signals from weekly RSI extremes
        rsi_overbought = rsi_1w_aligned[i] >= WEEKLY_RSI_OVERBOUGHT
        rsi_oversold = rsi_1w_aligned[i] <= WEEKLY_RSI_OVERSOLD
        
        # Generate signals
        if position == 0:
            if rsi_oversold and inside_day:
                # Bullish reversal: oversold weekly RSI + daily rejection of lower prices
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif rsi_overbought and inside_day:
                # Bearish reversal: overbought weekly RSI + daily rejection of higher prices
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals