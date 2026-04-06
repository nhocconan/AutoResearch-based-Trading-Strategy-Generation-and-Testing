#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h VWAP deviation with volume confirmation and ATR stoploss.
# Uses 12h VWAP as mean reversion target - price tends to revert to VWAP in ranging markets.
# Volume confirmation filters out weak moves. Designed for 75-200 trades over 4 years.
# Works in bull (buying dips to VWAP) and bear (selling rallies to VWAP) markets.
# Target: 100-180 total trades, 0.25 position size, max DD < -50%.

name = "exp_13753_4h_vwap_dev_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters - tuned for moderate trade frequency
VWAP_DEV_PERIOD = 20
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP over a period"""
    typical_price = (high + low + close) / 3.0
    vw = typical_price * volume
    cum_vw = pd.Series(vw).rolling(window=period, min_periods=period).sum().values
    cum_vol = pd.Series(volume).rolling(window=period, min_periods=period).sum().values
    vwap = np.divide(cum_vw, cum_vol, out=np.zeros_like(cum_vw), where=cum_vol!=0)
    return vwap

def calculate_vwap_deviation(price, vwap):
    """Calculate percentage deviation from VWAP"""
    return (price - vwap) / vwap * 100.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for VWAP calculation ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h VWAP
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    vwap_12h = calculate_vwap(high_12h, low_12h, close_12h, volume_12h, VWAP_DEV_PERIOD)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Calculate 4h indicators for entries
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss (using 4h data)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # VWAP deviation
    vwap_dev = calculate_vwap_deviation(close, vwap_12h_aligned)
    
    # Volume MA for 4h
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_DEV_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap_12h_aligned[i]) or np.isnan(vwap_dev[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # VWAP mean reversion signals
        long_signal = volume_ok and vwap_dev[i] < -1.0  # Price below VWAP
        short_signal = volume_ok and vwap_dev[i] > 1.0   # Price above VWAP
        
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
            # Exit long when price returns to VWAP
            if vwap_dev[i] >= 0.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price returns to VWAP
            if vwap_dev[i] <= 0.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals