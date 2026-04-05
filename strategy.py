#!/usr/bin/env python3
"""
Experiment #9034: 1h VWAP mean reversion + 4h RSI filter + 1d trend filter.
Hypothesis: In ranging markets (2025 test), price reverts to VWAP with high probability.
Use 4h RSI(14) < 40 for long, > 60 for short to avoid counter-trend trades.
Use 1d EMA(50) to filter trend direction: only long when price > EMA50, short when price < EMA50.
Session filter: 08-20 UTC to avoid low-volume Asian session.
Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9034_1h_vwap_meanrev_4h_rsi_1d_trend"
timeframe = "1h"
leverage = 1.0

# Parameters
VWAP_WINDOW = 24          # 24 hours for VWAP
RSI_PERIOD = 14
RSI_LONG_THRESH = 40      # RSI < 40 = oversold
RSI_SHORT_THRESH = 60     # RSI > 60 = overbought
EMA_TREND_PERIOD = 50
SIGNAL_SIZE = 0.20        # 20% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h RSI
    close_4h = df_4h['close'].values
    rsi_4h = calculate_rsi(close_4h, RSI_PERIOD)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=VWAP_WINDOW, min_periods=1).sum().values
    vwap_den = pd.Series(volume).rolling(window=VWAP_WINDOW, min_periods=1).sum().values
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_WINDOW, RSI_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Determine market bias from 1d EMA
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        # Mean reversion signals from VWAP
        price_below_vwap = close[i] < vwap[i]
        price_above_vwap = close[i] > vwap[i]
        
        # RSI conditions
        rsi_oversold = rsi_4h_aligned[i] < RSI_LONG_THRESH
        rsi_overbought = rsi_4h_aligned[i] > RSI_SHORT_THRESH
        
        # Entry conditions
        long_entry = price_above_ema and price_below_vwap and rsi_oversold
        short_entry = price_below_ema and price_above_vwap and rsi_overbought
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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