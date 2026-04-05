#!/usr/bin/env python3
"""
Experiment #8051: 6-hour Camarilla Pivot with 1-day Direction and Volume Confirmation.
Hypothesis: At 6h, price rejecting Camarilla R3/S3 levels with volume confirmation 
and aligned with 1d trend (price above/below 1d VWAP) captures mean-reversion bounces 
in ranging markets and breakout continuations in trending markets. Uses fade at R3/S3 
and continuation at R4/S4. Target: 75-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8051_6h_camarilla1d_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1 / 4  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
VWAP_PERIOD = 14
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels for given HLC"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close  # fallback
    
    pivot = (high + low + close) / 3
    r4 = close + CAMARILLA_MULTIPLIER * range_val * 1.1 / 2  # R4
    r3 = close + CAMARILLA_MULTIPLIER * range_val * 1.1 / 4  # R3
    s3 = close - CAMARILLA_MULTIPLIER * range_val * 1.1 / 4  # S3
    s4 = close - CAMARILLA_MULTIPLIER * range_val * 1.1 / 2  # S4
    return r3, r4, s3, s4

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP using typical price * volume"""
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).rolling(window=period, min_periods=1).sum()
    vwap_den = pd.Series(volume).rolling(window=period, min_periods=1).sum()
    vwap = vwap_num / vwap_den
    return vwap.replace(0, np.nan).fillna(method='ffill').values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP for trend bias
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d, VWAP_PERIOD)
    price_vs_vwap = np.where(close_1d > vwap_1d, 1, -1)  # 1=bullish bias, -1=bearish bias
    price_vs_vwap_aligned = align_htf_to_ltf(prices, df_1d, price_vs_vwap)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP for entry timing
    vwap = calculate_vwap(high, low, close, volume, VWAP_PERIOD)
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_vwap_aligned[i]):
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
        
        # Determine market bias from 1d VWAP
        bull_bias = price_vs_vwap_aligned[i] == 1   # 1d close above VWAP
        bear_bias = price_vs_vwap_aligned[i] == -1  # 1d close below VWAP
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Calculate Camarilla levels for current 6h bar
        r3, r4, s3, s4 = calculate_camarilla_levels(high[i], low[i], close[i])
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if bull_bias and volume_confirmed:
            # In bullish 1d context: look for bounces at S3/S4 or breakouts above R4
            if close[i] <= s3 and low[i] <= s3:  # touched or went below S3
                long_entry = True  # fade at S3
            elif close[i] >= r4 and high[i] >= r4:  # broke above R4
                long_entry = True  # continuation breakout
                
        if bear_bias and volume_confirmed:
            # In bearish 1d context: look for bounces at R3/R4 or breakouts below S4
            if close[i] >= r3 and high[i] >= r3:  # touched or went above R3
                short_entry = True  # fade at R3
            elif close[i] <= s4 and low[i] <= s4:  # broke below S4
                short_entry = True  # continuation breakdown
        
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