#!/usr/bin/env python3
"""
Experiment #8251: 6-hour Camarilla pivot + 1-day regime filter.
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h 
combined with 1-day trend regime (price above/below 1-day EMA50) provides high-probability 
entries. In bull regime (price > EMA50): fade at S3/R3 with stop at S4/R4. In bear regime 
(price < EMA50): fade at R3/S3 with stop at R4/S4. Uses volume confirmation to filter 
low-probability setups. Targets 50-150 trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8251_6h_camarilla1d_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 10
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close  # fallback
    
    c = close
    h = high
    l = low
    
    # Camarilla levels
    r4 = c + (range_val * 1.1 / 2)
    r3 = c + (range_val * 1.1 / 4)
    s3 = c - (range_val * 1.1 / 4)
    s4 = c - (range_val * 1.1 / 2)
    
    return r4, r3, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA for regime filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    bull_regime = close_1d > ema_1d  # True = bull regime, False = bear regime
    bull_regime_aligned = align_htf_to_ltf(prices, df_1d, bull_regime)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla levels (based on previous bar)
    r4_vals = np.full(n, np.nan)
    r3_vals = np.full(n, np.nan)
    s3_vals = np.full(n, np.nan)
    s4_vals = np.full(n, np.nan)
    
    for i in range(1, n):
        r4, r3, s3, s4 = calculate_camarilla(high[i-1], low[i-1], close[i-1])
        r4_vals[i] = r4
        r3_vals[i] = r3
        s3_vals[i] = s3
        s4_vals[i] = s4
    
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
    start = max(CAMARILLA_LOOKBACK + 1, VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(bull_regime_aligned[i]):
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
        
        # Determine regime
        is_bull_regime = bull_regime_aligned[i]
        is_bear_regime = not bull_regime_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Skip if not volume confirmed
        if not volume_confirmed:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Fade at S3/R3 with stop at S4/R4 based on regime
        if is_bull_regime:
            # In bull regime: fade at S3 (mean reversion up), stop at S4
            long_entry = (close[i] <= s3_vals[i]) and (i-1 >= 0) and (close[i-1] > s3_vals[i-1])
            long_exit = close[i] >= s4_vals[i]
            
            if position == 0 and long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = s4_vals[i]
            elif position == 1 and long_exit:
                signals[i] = 0.0
                position = 0
            elif position == 1:
                signals[i] = SIGNAL_SIZE
            else:
                signals[i] = 0.0
                
        else:  # bear regime
            # In bear regime: fade at R3 (mean reversion down), stop at R4
            short_entry = (close[i] >= r3_vals[i]) and (i-1 >= 0) and (close[i-1] < r3_vals[i-1])
            short_exit = close[i] <= r4_vals[i]
            
            if position == 0 and short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = r4_vals[i]
            elif position == -1 and short_exit:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = 0.0
    
    return signals