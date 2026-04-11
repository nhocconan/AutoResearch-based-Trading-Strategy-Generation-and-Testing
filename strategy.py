#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX momentum with 1w volume regime filter and 1d ATR stoploss
# - Long: TRIX(12) crosses above zero (bullish momentum) AND 1w volume > 1.5x 20-period average (institutional participation)
# - Short: TRIX(12) crosses below zero (bearish momentum) AND 1w volume > 1.5x 20-period average
# - Exit: TRIX returns to zero line OR ATR-based stop (2.5 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - TRIX is a triple-smoothed EMA momentum oscillator that reduces noise and false signals
# - Volume regime filter ensures we only trade when there is genuine market participation
# - ATR stoploss adapts to volatility conditions
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "12h_1d_1w_trix_volume_atrstop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w volume SMA(20) for regime filter
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Load 1d data ONCE before loop for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return signals
    
    # Pre-compute 1d TRIX(12,9,9) - triple smoothed EMA
    close_1d = df_1d['close'].values
    
    # First EMA
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Second EMA of first EMA
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Third EMA of second EMA
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # TRIX = percentage change of third EMA
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Handle first value
    trix[0] = 0.0
    
    # Align 1d TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Pre-compute 1d TRIX previous value for crossover detection
    trix_prev = np.roll(trix_aligned, 1)
    trix_prev[0] = trix_aligned[0]  # first value
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_prev[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # TRIX values
        trix_current = trix_aligned[i]
        trix_previous = trix_prev[i]
        
        # Volume regime filter: current 12h volume > 1.5x 1w volume SMA(20)
        vol_regime = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long entry: TRIX crosses above zero (bullish momentum) with volume regime
        if trix_previous <= 0 and trix_current > 0 and vol_regime:
            enter_long = True
        
        # Short entry: TRIX crosses below zero (bearish momentum) with volume regime
        if trix_previous >= 0 and trix_current < 0 and vol_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if TRIX returns to zero or ATR-based stop
            exit_long = (trix_current <= 0) or (close_price <= entry_price - 2.5 * atr_14[i])
        elif position == -1:
            # Exit short if TRIX returns to zero or ATR-based stop
            exit_short = (trix_current >= 0) or (close_price >= entry_price + 2.5 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals