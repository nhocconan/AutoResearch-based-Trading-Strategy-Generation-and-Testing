#!/usr/bin/env python3
"""
Experiment #8231: 6-hour ADX Trend Strength with 1-day ATR Range Filter
Hypothesis: Strong trends (ADX > 25) combined with low volatility environments 
(ATR ratio < 0.8) produce high-quality breakout entries. The 1-day ATR ratio 
filters for compression before expansion, while 6-hour ADX confirms trend 
strength. This combination works in both bull and bear markets by capturing 
sustained moves after periods of consolidation, avoiding whipsaw in ranging 
conditions. Target: 75-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8231_6h_adx25_atr_ratio_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ATR_RATIO_PERIOD = 10
ATR_RATIO_LONG_THRESHOLD = 0.8
ATR_RATIO_SHORT_THRESHOLD = 0.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR and its 10-period SMA for ratio
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=ATR_RATIO_PERIOD, min_periods=ATR_RATIO_PERIOD).mean().values
    # Avoid division by zero
    atr_ratio_1d = np.where(atr_ma_1d > 0, atr_1d / atr_ma_1d, 1.0)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX calculation
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) > 0, dx, 0.0)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # ATR for risk management (separate from ADX calculation)
    atr_risk = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD * 2, ATR_RATIO_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(atr_ratio_1d_aligned[i]):
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
        
        # Determine conditions
        strong_trend = adx[i] > ADX_THRESHOLD
        low_volatility = atr_ratio_1d_aligned[i] < ATR_RATIO_LONG_THRESHOLD
        
        # Direction from DI crossover
        bullish_cross = plus_di[i] > minus_di[i]
        bearish_cross = minus_di[i] > plus_di[i]
        
        # Entry conditions
        long_entry = strong_trend and low_volatility and bullish_cross
        short_entry = strong_trend and low_volatility and bearish_cross
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_risk[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_risk[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals