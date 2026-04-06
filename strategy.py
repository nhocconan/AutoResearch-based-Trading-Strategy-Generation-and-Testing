#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly RSI mean-reversion with monthly trend filter on 12h timeframe
# Uses weekly RSI extremes (>80 or <20) for mean reversion, confirmed by monthly SMA trend.
# Works in bull/bear because mean reversion captures overextended moves, and trend filter
# avoids fighting the major direction. Target: 50-120 trades over 4 years (12-30/year).

name = "exp_13008_12h_weekly_rsi_meanrev_monthly_trend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
RSI_OVERSOLD = 20
SMA_TREND_PERIOD = 50
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

def calculate_sma(close, period):
    """Calculate SMA"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    if n < 100:
        return np.zeros(n)
    
    # Load weekly and monthly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    df_monthly = get_htf_data(prices, '1M')
    
    # Calculate weekly RSI
    close_w = df_weekly['close'].values
    rsi_w = calculate_rsi(close_w, RSI_PERIOD)
    
    # Calculate monthly SMA for trend filter
    close_m = df_monthly['close'].values
    sma_m = calculate_sma(close_m, SMA_TREND_PERIOD)
    
    # Align to 12h timeframe
    rsi_w_aligned = align_htf_to_ltf(prices, df_weekly, rsi_w)
    sma_m_aligned = align_htf_to_ltf(prices, df_monthly, sma_m)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, SMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(rsi_w_aligned[i]) or np.isnan(sma_m_aligned[i]):
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
        
        # Mean reversion signals from weekly RSI
        rsi_oversold = rsi_w_aligned[i] < RSI_OVERSOLD
        rsi_overbought = rsi_w_aligned[i] > RSI_OVERBOUGHT
        
        # Trend filter from monthly SMA
        price_above_sma = close[i] > sma_m_aligned[i]
        price_below_sma = close[i] < sma_m_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long when RSI oversold and price above monthly SMA (uptrend)
            if rsi_oversold and price_above_sma:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short when RSI overbought and price below monthly SMA (downtrend)
            elif rsi_overbought and price_below_sma:
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