#!/usr/bin/env python3
"""
Experiment #7927: 6-hour price action with 1d pivot confluence and volume confirmation.
Hypothesis: Price breaking beyond daily pivot levels (S1/S2/S3 or R1/R2/R3) on 6h with volume > 1.5x 20-period MA 
and aligned 1d trend (price above/below daily EMA50) captures reversal or continuation moves. 
Daily pivot levels provide institutional reference points that work in both bull and bear markets, 
while the 1d EMA filter ensures we trade with the higher timeframe trend. 
Target: 50-150 total trades over 4 years (12-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7927_6h_pivot1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_PERIOD = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points and support/resistance levels."""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate daily pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points for each day
    pivot_vals = np.full(len(close_1d), np.nan)
    r1_vals = np.full(len(close_1d), np.nan)
    r2_vals = np.full(len(close_1d), np.nan)
    r3_vals = np.full(len(close_1d), np.nan)
    s1_vals = np.full(len(close_1d), np.nan)
    s2_vals = np.full(len(close_1d), np.nan)
    s3_vals = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i >= PIVOT_PERIOD:  # Need previous day data
            pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(
                high_1d[i-1], low_1d[i-1], close_1d[i-1]
            )
            pivot_vals[i] = pivot
            r1_vals[i] = r1
            r2_vals[i] = r2
            r3_vals[i] = r3
            s1_vals[i] = s1
            s2_vals[i] = s2
            s3_vals[i] = s3
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    target_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - price breaking through pivot levels
        # Long: price breaks above R1 (in bullish bias) or S2/S3 (mean reversion in bearish bias)
        # Short: price breaks below S1 (in bearish bias) or R2/R3 (mean reversion in bullish bias)
        long_breakout = (
            (bull_bias and close[i] > r1_aligned[i-1]) or  # Continuation in bullish trend
            (not bull_bias and close[i] > s2_aligned[i-1])  # Mean reversion from oversold
        )
        short_breakout = (
            (bear_bias and close[i] < s1_aligned[i-1]) or  # Continuation in bearish trend
            (not bear_bias and close[i] < r2_aligned[i-1])  # Mean reversion from overbought
        )
        
        # Entry conditions
        long_entry = long_breakout and volume_confirmed
        short_entry = short_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals