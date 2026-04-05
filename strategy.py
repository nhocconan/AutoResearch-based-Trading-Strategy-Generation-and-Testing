#!/usr/bin/env python3
"""
Experiment #8252: 12-hour Donchian channel breakout with 1-day trend filter.
Hypothesis: Breakouts above Donchian(20) high or below Donchian(20) low on 12h,
filtered by 1-day EMA trend (price above/below EMA50), with volume confirmation.
In bull trend (price > EMA50): long breakouts only. In bear trend (price < EMA50): short breakdowns only.
Uses ATR-based stop loss. Targets 50-150 trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8252_12h_donchian20_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    bull_trend = close_1d > ema_1d  # True = bull trend, False = bear trend
    bull_trend_aligned = align_htf_to_ltf(prices, df_1d, bull_trend)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (based on previous bar to avoid look-ahead)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(DONCHIAN_PERIOD, n):
        # Use data up to i-1 to calculate bands for bar i
        upper_band[i] = np.max(high[i-DONCHIAN_PERIOD:i])
        lower_band[i] = np.min(low[i-DONCHIAN_PERIOD:i])
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD + 1, VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(bull_trend_aligned[i]):
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
        
        # Determine trend
        is_bull_trend = bull_trend_aligned[i]
        is_bear_trend = not bull_trend_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Skip if not volume confirmed
        if not volume_confirmed:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Breakout logic based on trend
        if is_bull_trend:
            # In bull trend: look for long breakouts above upper band
            long_entry = (i-1 >= 0) and (close[i] > upper_band[i]) and (close[i-1] <= upper_band[i])
            
            if position == 0 and long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif position == 1:
                signals[i] = SIGNAL_SIZE
            else:
                signals[i] = 0.0
                
        else:  # bear trend
            # In bear trend: look for short breakdowns below lower band
            short_entry = (i-1 >= 0) and (close[i] < lower_band[i]) and (close[i-1] >= lower_band[i])
            
            if position == 0 and short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif position == -1:
                signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = 0.0
    
    return signals