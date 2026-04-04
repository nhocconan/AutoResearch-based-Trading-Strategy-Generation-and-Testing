#!/usr/bin/env python3
"""
exp_6570_1d_donchian20_1w_hma_vol_v1
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
Uses 1d primary timeframe to minimize fee drag (target: 30-100 total trades over 4 years).
1w HMA provides smooth trend direction that works in both bull and bear markets.
Volume confirmation ensures breakouts have conviction. Discrete sizing (0.25) minimizes fee churn.
ATR-based stoploss limits downside risk.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6570_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final HMA
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = 0  # First TR is undefined
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    return atr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA
    hma_1w = calculate_hma(df_1w['close'].values, HMA_PERIOD)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w ATR for dynamic stoploss
    atr_1w = calculate_atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, ATR_PERIOD)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, HMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            stop_price = entry_price - ATR_STOP_MULT * entry_atr
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            stop_price = entry_price + ATR_STOP_MULT * entry_atr
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Determine trend direction from 1w HMA
        # Price above HMA: bullish trend (favor longs)
        # Price below HMA: bearish trend (favor shorts)
        price_above_hma = close[i] > hma_1w_aligned[i]
        
        # Long conditions: 
        # 1. Break above Donchian HIGH (breakout)
        # 2. Volume confirmation
        # 3. Bullish trend filter (price > HMA)
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        long_trend = price_above_hma
        
        # Short conditions:
        # 1. Break below Donchian LOW (breakdown)
        # 2. Volume confirmation
        # 3. Bearish trend filter (price < HMA)
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        short_trend = not price_above_hma  # price below HMA
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume and long_trend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                entry_atr = atr_1w_aligned[i]
                bars_since_entry = 0
            elif short_breakout and short_volume and short_trend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                entry_atr = atr_1w_aligned[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals