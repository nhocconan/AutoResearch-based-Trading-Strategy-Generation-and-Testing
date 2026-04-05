#!/usr/bin/env python3
"""
Experiment #9694: 1h Hull Moving Average Trend + Volume Spike + Session Filter.
Hypothesis: HMA(21) on 4h provides reliable trend direction, while volume spikes on 1h 
provide timely entry signals during active sessions (08-20 UTC). This combination 
works in bull markets (follow 4h trend) and bear markets (fade false breakouts at 
extremes). Targets 60-150 total trades over 4 years (15-38/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9694_1h_hma_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean()
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend direction)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, HMA_PERIOD)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d trend filter for stronger bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(HMA_PERIOD, 20, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma_50_1d_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filters
        hma_uptrend = hma_4h_aligned[i] > hma_4h_aligned[i-1] if i > 0 else False
        hma_downtrend = hma_4h_aligned[i] < hma_4h_aligned[i-1] if i > 0 else False
        price_above_1dsma = close[i] > sma_50_1d_aligned[i] if not np.isnan(sma_50_1d_aligned[i]) else False
        price_below_1dsma = close[i] < sma_50_1d_aligned[i] if not np.isnan(sma_50_1d_aligned[i]) else False
        
        # Entry conditions
        # Long: 4h HMA uptrend + price above 1d SMA + volume spike
        long_entry = hma_uptrend and price_above_1dsma and volume_spike
        # Short: 4h HMA downtrend + price below 1d SMA + volume spike
        short_entry = hma_downtrend and price_below_1dsma and volume_spike
        
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