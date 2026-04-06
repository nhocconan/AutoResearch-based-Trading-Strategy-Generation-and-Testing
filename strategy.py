#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price closing above/below weekly 200-bar SMA with volume confirmation.
# Weekly SMA acts as strong support/resistance. Price closing above/below indicates regime change.
# Volume confirmation ensures institutional participation. Works in bull markets (buy breaks above weekly SMA) 
# and bear markets (sell breaks below weekly SMA). Target: 15-25 trades/year.

name = "exp_13615_6s_wk200ma_close_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WEEKLY_SMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_sma(close, period):
    """Calculate SMA"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for SMA filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly SMA
    close_weekly = df_weekly['close'].values
    sma_weekly = calculate_sma(close_weekly, WEEKLY_SMA_PERIOD)
    sma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_SMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(sma_weekly_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Price relative to weekly SMA
        price_above_sma = close[i] > sma_weekly_aligned[i]
        price_below_sma = close[i] < sma_weekly_aligned[i]
        
        # Entry signals: close crosses weekly SMA with volume
        # Avoid lookback by checking current and previous values
        if i > 0 and not np.isnan(sma_weekly_aligned[i-1]):
            sma_prev = sma_weekly_aligned[i-1]
            close_prev = close[i-1]
            close_curr = close[i]
            
            # Long signal: price closes above weekly SMA from below with volume
            long_signal = volume_ok and (close_prev <= sma_prev) and (close_curr > sma_weekly_aligned[i])
            
            # Short signal: price closes below weekly SMA from above with volume
            short_signal = volume_ok and (close_prev >= sma_prev) and (close_curr < sma_weekly_aligned[i])
        else:
            long_signal = False
            short_signal = False
        
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
            # Exit long when price closes below weekly SMA
            if close[i] < sma_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price closes above weekly SMA
            if close[i] > sma_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals