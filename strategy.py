#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend filter with weekly RSI confirmation and volume spike for breakouts.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# Weekly RSI ensures alignment with higher timeframe momentum.
# Volume spikes confirm genuine breakout strength.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.
# Works in bull markets by catching breakouts with momentum, in bear markets by fading overextended moves.

name = "exp_13278_1d_kama10_rsi14_vol_spike_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_ER_FAST = 2
KAMA_ER_SLOW = 30
KAMA_PERIOD = 10
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_kama(close, er_fast, er_slow, period):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, period))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # Calculate weekly RSI for trend filter
    close_1w = df_1w['close'].values
    rsi_1w = calculate_rsi(close_1w, RSI_PERIOD)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA
    kama = calculate_kama(close, KAMA_ER_FAST, KAMA_ER_SLOW, KAMA_PERIOD)
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_PERIOD, RSI_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if RSI not available
        if np.isnan(rsi_1w_aligned[i]):
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
        
        # Volume spike confirmation
        volume_ma = np.mean(volume[max(0, i-19):i+1])  # 20-period volume MA
        volume_spike = volume[i] > (volume_ma * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma) else False
        
        # Trend filter: price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Weekly RSI filter
        rsi_overbought = rsi_1w_aligned[i] > RSI_OVERBOUGHT
        rsi_oversold = rsi_1w_aligned[i] < RSI_OVERSOLD
        
        # Entry signals
        if position == 0:
            # Long: price above KAMA, not overbought weekly RSI, volume spike
            if price_above_kama and not rsi_overbought and volume_spike:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: price below KAMA, not oversold weekly RSI, volume spike
            elif price_below_kama and not rsi_oversold and volume_spike:
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