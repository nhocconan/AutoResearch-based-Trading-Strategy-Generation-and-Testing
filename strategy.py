#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R (from daily) with volume confirmation and
# daily EMA trend filter. Williams %R identifies overbought/oversold conditions;
# reversals with volume indicate institutional interest. Daily EMA ensures alignment
# with higher timeframe momentum to avoid counter-trend trades. Target: 50-150 total trades over 4 years.
# Works in bull markets (reversals from oversold) and bear markets (reversals from overbought).

name = "exp_13376_12h_williamsr_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14  # Williams %R period (daily)
EMA_PERIOD = 20       # Daily EMA
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    # Handle division by zero
    wr = wr.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    return wr.values

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Williams %R
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    williams_r = calculate_williams_r(high_1d, low_1d, close_1d, WILLIAMS_PERIOD)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate daily EMA for trend filter
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Williams %R or EMA not available
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Williams %R reversal signals
        # Long: Williams %R crosses above -80 (oversold) with volume and uptrend
        # Short: Williams %R crosses below -20 (overbought) with volume and downtrend
        williams_prev = williams_r_aligned[i-1]
        williams_curr = williams_r_aligned[i]
        
        # Avoid using values if previous is NaN
        if np.isnan(williams_prev):
            williams_prev = williams_curr
        
        reversal_up = volume_ok and uptrend and (williams_prev <= -80) and (williams_curr > -80)
        reversal_down = volume_ok and downtrend and (williams_prev >= -20) and (williams_curr < -20)
        
        # Generate signals
        if position == 0:
            if reversal_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif reversal_down:
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