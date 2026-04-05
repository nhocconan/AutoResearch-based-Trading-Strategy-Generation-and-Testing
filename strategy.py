#!/usr/bin/env python3
"""
Experiment #8835: 6h Donchian(20) breakout + 1d VWAP filter + volume confirmation + ATR stoploss.
Hypothesis: 6h timeframe balances trade frequency and trend capture. Donchian breakouts capture momentum,
while 1d VWAP acts as dynamic support/resistance to avoid false breakouts. Volume confirms institutional
participation. Works in both bull (breakouts with volume) and bear (fades at VWAP during low volume).
Targets 75-200 total trades over 4 years (~19-50/year) to balance frequency and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8835_6h_donchian20_1d_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VWAP_PERIOD = 24  # 1 day of 6h bars
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP using typical price"""
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=period, min_periods=period).sum()
    vwap_den = pd.Series(volume).rolling(window=period, min_periods=period).sum()
    vwap = (vwap_num / vwap_den).values
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d, VWAP_PERIOD)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VWAP_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(vwap_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # VWAP position: above = bullish bias, below = bearish bias
        price_vs_vwap = close[i] > vwap_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: breakout in direction of VWAP bias with volume
        long_entry = price_vs_vwap and long_breakout and volume_confirmed
        short_entry = (not price_vs_vwap) and short_breakout and volume_confirmed
        
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
</x>