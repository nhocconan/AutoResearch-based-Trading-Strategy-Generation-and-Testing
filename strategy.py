#!/usr/bin/env python3
"""
Experiment #7851: 6-hour Camarilla pivot reversal with daily trend filter.
Hypothesis: Price rejecting at Camarilla R3/S3 levels on 6h with 1d trend alignment captures reversals in both trending and ranging markets. The daily trend filter ensures we only take reversals in the direction of higher timeframe momentum, avoiding counter-trend traps. Targets 50-150 trades over 4 years with controlled risk via fixed stop loss.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7851_6h_camarilla_reversal_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Trend bias: above EMA = bullish, below EMA = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
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
    start = max(EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_bias_1d_aligned[i]):
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
        
        # Determine market bias from 1d EMA
        bull_bias = trend_bias_1d_aligned[i] == 1   # 1d close above EMA
        bear_bias = trend_bias_1d_aligned[i] == -1  # 1d close below EMA
        
        # Calculate Camarilla levels from previous day's price action
        # Use 1d data shifted by 1 to avoid look-ahead
        if i >= 24:  # Need at least 1 day of 6h data (24 bars)
            # Get previous day's OHLC from 1d data
            day_index = i // 24 - 1  # Previous completed day
            if day_index >= 0 and day_index < len(df_1d):
                ph = df_1d['high'].iloc[day_index]
                pl = df_1d['low'].iloc[day_index]
                pc = df_1d['close'].iloc[day_index]
                
                # Calculate Camarilla levels
                r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla(ph, pl, pc)
                
                # Reversal conditions at R3/S3 with trend alignment
                # Long when price rejects S3 in uptrend
                long_rejection = (low[i] <= s3 * 1.001 and close[i] > s3) and bull_bias
                # Short when price rejects R3 in downtrend
                short_rejection = (high[i] >= r3 * 0.999 and close[i] < r3) and bear_bias
                
                # Entry conditions
                long_entry = long_rejection
                short_entry = short_rejection
                
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
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        else:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals