#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend filter with weekly RSI momentum and volume confirmation
# Uses weekly trend direction (KAMA) for bias, daily RSI for entry timing, and volume surge for confirmation
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Target: ~60 total trades, 0.25 position size, max DD < -50%

name = "exp_13730_1d_kama_rsi_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters - tuned for low trade frequency
KAMA_PERIOD = 10
KAMA_FAST = 2
KAMA_SLOW = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_kama(close, period, fast, slow):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, period))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
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
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly KAMA for trend filter
    close_weekly = df_weekly['close'].values
    kama_weekly = calculate_kama(close_weekly, KAMA_PERIOD, KAMA_FAST, KAMA_SLOW)
    kama_weekly_aligned = align_htf_to_ltf(prices, df_weekly, kama_weekly)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Daily RSI for entry timing
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(kama_weekly_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from weekly KAMA
        above_kama = close[i] > kama_weekly_aligned[i]
        below_kama = close[i] < kama_weekly_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Generate signals
        if position == 0:
            # Long: weekly uptrend + daily oversold + volume surge
            if above_kama and rsi_oversold and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: weekly downtrend + daily overbought + volume surge
            elif below_kama and rsi_overbought and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on weekly trend reversal or RSI overbought
            if below_kama or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on weekly trend reversal or RSI oversold
            if above_kama or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals