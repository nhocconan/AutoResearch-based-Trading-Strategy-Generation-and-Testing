#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12684_1d_wk_volatility_breakout_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
VOLATILITY_PERIOD = 20
VOLATILITY_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLDING_PERIOD = 15  # max 15 days

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Calculate weekly ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, VOLATILITY_PERIOD)
    
    # Calculate volatility ratio (current vs historical)
    atr_ratio = atr_1w / (pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values + 1e-10)
    
    # Align volatility ratio to daily timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_daily = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_bar = 0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, 50) + 1
    
    for i in range(start, n):
        # Skip if volatility ratio not available
        if np.isnan(atr_ratio_aligned[i]):
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
        
        # Check max holding period
        if position != 0 and (i - entry_bar) >= MAX_HOLDING_PERIOD:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volatility filter: only trade when volatility is expanding
        vol_expanding = atr_ratio_aligned[i] > VOLATILITY_MULTIPLIER
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions: price breaks ATR-based bands
        upper_band = close[i-1] + (VOLATILITY_MULTIPLIER * atr_daily[i-1])
        lower_band = close[i-1] - (VOLATILITY_MULTIPLIER * atr_daily[i-1])
        
        breakout_up = vol_expanding and volume_ok and close[i] > upper_band
        breakout_down = vol_expanding and volume_ok and close[i] < lower_band
        
        # Entry conditions
        long_entry = breakout_up
        short_entry = breakout_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                entry_bar = i
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_daily[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                entry_bar = i
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_daily[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals