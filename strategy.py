#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR breakout with 12h mean reversion filter
# Works in bull/bear because breakouts capture strong moves, while 12h RSI mean reversion
# filters out false breakouts in choppy markets. Uses volatility expansion (ATR) to
# detect genuine breakouts and RSI to avoid overextended moves. Target: 75-150 total trades.

name = "exp_13019_6h_atr_breakout_12h_rsi_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SIGNAL_SIZE = 0.25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h RSI
    close_12h = df_12h['close'].values
    rsi_12h = calculate_rsi(close_12h, RSI_PERIOD)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ATR_PERIOD, VOLUME_MA_PERIOD, RSI_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if RSI not available
        if np.isnan(rsi_12h_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # ATR breakout conditions
        breakout_up = volume_ok and (high[i] > close[i-1] + ATR_MULTIPLIER * atr[i-1])
        breakout_down = volume_ok and (low[i] < close[i-1] - ATR_MULTIPLIER * atr[i-1])
        
        # 12h RSI mean reversion filter
        rsi_not_extreme = (rsi_12h_aligned[i] > RSI_OVERSOLD) and (rsi_12h_aligned[i] < RSI_OVERBOUGHT)
        rsi_oversold = rsi_12h_aligned[i] < RSI_OVERSOLD
        rsi_overbought = rsi_12h_aligned[i] > RSI_OVERBOUGHT
        
        # Generate signals
        if position == 0:
            # Long: breakout up with RSI not overbought (or oversold for mean reversion)
            if breakout_up and (rsi_not_extreme or rsi_oversold):
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_MULTIPLIER * atr[i])
            # Short: breakout down with RSI not oversold (or overbought for mean reversion)
            elif breakout_down and (rsi_not_extreme or rsi_overbought):
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals