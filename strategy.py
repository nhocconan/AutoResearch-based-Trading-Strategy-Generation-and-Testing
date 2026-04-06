#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion filtered by 4h and 1d trend.
# RSI(14) < 30 and > 70 identifies overextended moves. In strong trends (4h/1d EMA200),
# these reversals have high probability. Volume confirmation filters weak signals.
# Works in bull/bear because mean reversion occurs in all regimes, and trend filter
# ensures we trade with the higher timeframe momentum.
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and fees.

name = "exp_12974_1h_rsi_meanrev_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_TREND_FAST = 50
EMA_TREND_SLOW = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h and 1d EMAs for trend filter
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    ema_4h_fast = calculate_ema(close_4h, EMA_TREND_FAST)
    ema_4h_slow = calculate_ema(close_4h, EMA_TREND_SLOW)
    ema_1d_fast = calculate_ema(close_1d, EMA_TREND_FAST)
    ema_1d_slow = calculate_ema(close_1d, EMA_TREND_SLOW)
    
    # Align to 1h timeframe (already shifted by 1 in align_htf_to_ltf)
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    ema_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_fast)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_TREND_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA data not available
        if (np.isnan(ema_4h_fast_aligned[i]) or np.isnan(ema_4h_slow_aligned[i]) or
            np.isnan(ema_1d_fast_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Trend filter: 4h and 1d both bullish or bearish
        trend_4h_bullish = ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i]
        trend_4h_bearish = ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i]
        trend_1d_bullish = ema_1d_fast_aligned[i] > ema_1d_slow_aligned[i]
        trend_1d_bearish = ema_1d_fast_aligned[i] < ema_1d_slow_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Mean reversion signals
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Generate signals
        if position == 0:
            # Long: oversold RSI + bullish trend on both TFs + volume
            if rsi_oversold and trend_4h_bullish and trend_1d_bullish and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: overbought RSI + bearish trend on both TFs + volume
            elif rsi_overbought and trend_4h_bearish and trend_1d_bearish and volume_ok:
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