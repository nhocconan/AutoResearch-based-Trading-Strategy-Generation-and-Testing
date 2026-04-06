#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h RSI mean reversion with volume confirmation
# Works in bull/bear because Williams %R identifies overbought/oversold conditions,
# RSI filters for momentum strength, and volume confirms genuine moves.
# Target: 50-120 trades over 4 years (12-30/year) to balance opportunity and cost.

name = "exp_13019_6h_williamsr_12h_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = williams_r.fillna(-50)
    return williams_r.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Handle division by zero
    rsi = rsi.fillna(50)
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
    
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_R_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_R_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h RSI not available
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
        
        # Mean reversion signals from Williams %R
        williams_oversold = williams_r[i] <= WILLIAMS_R_OVERSOLD
        williams_overbought = williams_r[i] >= WILLIAMS_R_OVERBOUGHT
        
        # RSI momentum filter from 12h
        rsi_momentum_up = rsi_12h_aligned[i] > RSI_OVERBOUGHT
        rsi_momentum_down = rsi_12h_aligned[i] < RSI_OVERSOLD
        
        # Generate signals
        if position == 0:
            # Long: Williams %R oversold + 12h RSI not overbought + volume
            if williams_oversold and not rsi_momentum_up and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: Williams %R overbought + 12h RSI not oversold + volume
            elif williams_overbought and not rsi_momentum_down and volume_ok:
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