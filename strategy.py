#!/usr/bin/env python3
"""
Experiment #8531: 6h Camarilla pivot reversal with volume confirmation.
Hypothesis: Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout) provide
high-probability reversal/breakout zones. Combined with volume confirmation and
1d trend filter (close vs daily open), this should work in both bull and bear
markets by capturing mean reversion at extremes and momentum continuations.
Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity
and fee drag. Uses 6h timeframe for reduced noise vs lower timeframes.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8531_6h_camarilla_pivot_rev_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H/L/C from previous day
    R4 = close_1d + (high_1d - low_1d) * 1.500
    R3 = close_1d + (high_1d - low_1d) * 1.250
    S3 = close_1d - (high_1d - low_1d) * 1.250
    S4 = close_1d - (high_1d - low_1d) * 1.500
    
    # Daily trend filter: bullish if close > open, bearish if close < open
    open_1d = df_1d['open'].values
    daily_bull = close_1d > open_1d  # 1 if bullish day
    daily_bear = close_1d < open_1d  # 1 if bearish day
    
    # Align all HTF arrays to 6s timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    daily_bull_aligned = align_htf_to_ltf(prices, df_1d, daily_bull.astype(float))
    daily_bear_aligned = align_htf_to_ltf(prices, df_1d, daily_bear.astype(float))
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
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
        
        # Price action relative to Camarilla levels
        price = close[i]
        
        # Reversal zones: S3 (support) and R3 (resistance)
        near_S3 = price <= S3_aligned[i] * 1.002  # Within 0.2% of S3
        near_R3 = price >= R3_aligned[i] * 0.998  # Within 0.2% of R3
        
        # Breakout zones: S4 (breakdown) and R4 (breakout)
        below_S4 = price < S4_aligned[i]
        above_R4 = price > R4_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        # Long: reversal at S3 with volume OR breakout above R4 with volume + daily bull
        long_reversal = near_S3 and volume_confirmed
        long_breakout = above_R4 and volume_confirmed and daily_bull_aligned[i] == 1
        long_entry = long_reversal or long_breakout
        
        # Short: reversal at R3 with volume OR breakdown below S4 with volume + daily bear
        short_reversal = near_R3 and volume_confirmed
        short_breakout = below_S4 and volume_confirmed and daily_bear_aligned[i] == 1
        short_entry = short_reversal or short_breakout
        
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