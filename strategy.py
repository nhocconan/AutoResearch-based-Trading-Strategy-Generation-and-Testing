#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with volume confirmation and 12h EMA trend filter.
# Long when 4h close breaks above 12h Donchian upper band (20) with volume > 1.5x average and 12h EMA(50) rising.
# Short when 4h close breaks below 12h Donchian lower band (20) with volume > 1.5x average and 12h EMA(50) falling.
# Exit when price reverses to touch 12h EMA(50) or stop loss hit at 2.5x ATR.
# Works in bull (breaks above channel with uptrend) and bear (breaks below channel with downtrend).
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fees.

name = "exp_13753_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    return tr

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for Donchian and EMA ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian channel
    donch_upper_12h, donch_lower_12h = calculate_donchian(high_12h, low_12h, DONCHIAN_PERIOD)
    # 12h EMA for trend filter
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    
    # Align 12h indicators to 4h timeframe
    donch_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_upper_12h)
    donch_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_lower_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_upper_12h_aligned[i]) or np.isnan(donch_lower_12h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # EMA trend filter (rising/falling)
        ema_rising = ema_12h_aligned[i] > ema_12h_aligned[i-1]
        ema_falling = ema_12h_aligned[i] < ema_12h_aligned[i-1]
        
        # Entry signals based on 4h close breaking 12h Donchian
        long_signal = volume_ok and ema_rising and close[i] > donch_upper_12h_aligned[i]
        short_signal = volume_ok and ema_falling and close[i] < donch_lower_12h_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if price touches 12h EMA (trend weakness) or stop hit
            if close[i] <= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if price touches 12h EMA (trend weakness) or stop hit
            if close[i] >= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals