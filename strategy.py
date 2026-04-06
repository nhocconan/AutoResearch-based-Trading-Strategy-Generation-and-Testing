#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price action strategy combining Bollinger Band squeeze breakout 
# with RSI momentum and 1-day trend filter. Works in bull/bear markets because 
# Bollinger squeezes precede explosive moves, RSI filters for momentum strength, 
# and 1-day EMA ensures alignment with higher timeframe trend. Low trade frequency 
# (target: 80-150 total over 4 years) minimizes fee drag while capturing explosive moves.

name = "exp_13049_4h_bb_squeeze_rsi_1d_ema_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2.0
SQUEEZE_THRESHOLD = 0.03  # Bollinger Band width threshold for squeeze
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3

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

def calculate_bbands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper.values, lower.values, sma.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_middle = calculate_bbands(close, BB_PERIOD, BB_STD_DEV)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, RSI_PERIOD, EMA_PERIOD, ATR_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Bollinger Band squeeze condition
        squeeze_active = bb_width[i] < SQUEEZE_THRESHOLD if not np.isnan(bb_width[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # RSI momentum
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT if not np.isnan(rsi[i]) else False
        rsi_oversold = rsi[i] < RSI_OVERSOLD if not np.isnan(rsi[i]) else False
        
        # Breakout signals from Bollinger Bands
        breakout_up = squeeze_active and volume_ok and uptrend and rsi_oversold and (close[i] > bb_upper[i])
        breakout_down = squeeze_active and volume_ok and downtrend and rsi_overbought and (close[i] < bb_lower[i])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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