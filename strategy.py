#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour MACD histogram divergence with 1-week EMA trend filter and volume confirmation
# Works in bull/bear because MACD captures momentum shifts, EMA filter ensures trend alignment,
# and volume filters false signals. Target: 80-120 trades over 4 years (20-30/year).

name = "exp_12968_12h_macd_div_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_macd(close, fast, slow, signal):
    """Calculate MACD line, signal line, and histogram"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

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
    if n < 100:
        return np.zeros(n)
    
    # Load weekly EMA data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA
    ema_weekly = calculate_ema(df_weekly['close'].values, EMA_PERIOD)
    
    # Align weekly EMA to 12h timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate MACD
    macd_line, signal_line, macd_hist = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    
    # Calculate volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MACD_SLOW, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]):
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
        
        # MACD histogram divergence signals
        # Bullish divergence: price makes lower low, MACD makes higher low
        # Bearish divergence: price makes higher high, MACD makes lower high
        bullish_div = False
        bearish_div = False
        
        if i >= 2:
            # Bullish divergence: lower price low, higher MACD low
            if (close[i] < close[i-1] and close[i-1] < close[i-2] and 
                macd_hist[i] > macd_hist[i-1] and macd_hist[i-1] > macd_hist[i-2]):
                bullish_div = True
            # Bearish divergence: higher price high, lower MACD high
            elif (close[i] > close[i-1] and close[i-1] > close[i-2] and 
                  macd_hist[i] < macd_hist[i-1] and macd_hist[i-1] < macd_hist[i-2]):
                bearish_div = True
        
        # Additional filter: price relative to weekly EMA
        price_above_weekly_ema = close[i] > ema_weekly_aligned[i]
        price_below_weekly_ema = close[i] < ema_weekly_aligned[i]
        
        # Generate signals
        if position == 0:
            if bullish_div and price_above_weekly_ema and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_div and price_below_weekly_ema and volume_ok:
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