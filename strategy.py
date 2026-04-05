#!/usr/bin/env python3
"""
Experiment #10875: 6h Williams %R with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Williams %R captures overbought/oversold conditions, while weekly pivot provides long-term trend bias.
In bull markets: buy oversold dips in uptrend. In bear markets: sell rallies in downtrend.
Volume confirmation ensures institutional participation. Weekly pivot avoids whipsaws.
Target: 75-150 total trades over 4 years (19-38/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10875_6h_williamsr_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80
WEEKLY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend
    ema_weekly = calculate_ema(df_weekly['close'].values, WEEKLY_EMA_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_R_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_R_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]):
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
        
        # Williams %R conditions
        wr_oversold = williams_r[i] < WILLIAMS_R_OVERSOLD if not np.isnan(williams_r[i]) else False
        wr_overbought = williams_r[i] > WILLIAMS_R_OVERBOUGHT if not np.isnan(williams_r[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (weekly)
        uptrend_weekly = close[i] > ema_weekly_aligned[i]
        downtrend_weekly = close[i] < ema_weekly_aligned[i]
        
        # Entry conditions
        long_entry = wr_oversold and volume_ok and uptrend_weekly
        short_entry = wr_overbought and volume_ok and downtrend_weekly
        
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