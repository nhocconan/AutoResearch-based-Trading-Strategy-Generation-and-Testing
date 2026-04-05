#!/usr/bin/env python3
"""
Experiment #8855: 6h Donchian breakout + weekly pivot direction + volume confirmation
Hypothesis: Weekly pivot levels provide strong institutional reference points. 
Breaking above weekly R1 with volume in an uptrend (price > weekly VWAP) captures momentum with institutional participation.
Weekly context filters out noise, reducing false breakouts. 6h timeframe balances trade frequency and responsiveness.
Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag while maintaining statistical validity.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8855_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3.0
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points and VWAP
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    vol_1w = df_1w['volume'].values
    
    # Weekly pivot levels
    pw, r1w, r2w, s1w, s2w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Weekly VWAP
    vwap_1w = calculate_vwap(high_1w, low_1w, close_1w, vol_1w)
    
    # Price relative to weekly VWAP: above = bullish bias, below = bearish bias
    price_vs_vwap = np.where(close_1w > vwap_1w, 1, 
                     np.where(close_1w < vwap_1w, -1, 0))  # 1=bullish, -1=bearish, 0=at VWAP
    price_vs_vwap_aligned = align_htf_to_ltf(prices, df_1w, price_vs_vwap)
    
    # Weekly R1 level for breakout
    r1w_aligned = align_htf_to_ltf(prices, df_1w, r1w)
    
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
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_vwap_aligned[i]) or np.isnan(r1w_aligned[i]):
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
        
        # Determine market bias from weekly VWAP
        bull_bias = price_vs_vwap_aligned[i] == 1   # weekly price above VWAP
        bear_bias = price_vs_vwap_aligned[i] == -1  # weekly price below VWAP
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Weekly R1 breakout conditions (more specific)
        long_r1_breakout = close[i] > r1w_aligned[i-1] and close[i-1] <= r1w_aligned[i-1]
        short_s1_breakout = close[i] < s1w_aligned[i-1] and close[i-1] >= s1w_aligned[i-1]  # S1 breakdown
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: R1 breakout with volume in bullish weekly bias
        long_entry = bull_bias and long_r1_breakout and volume_confirmed
        # Entry conditions: S1 breakdown with volume in bearish weekly bias
        short_entry = bear_bias and short_s1_breakout and volume_confirmed
        
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