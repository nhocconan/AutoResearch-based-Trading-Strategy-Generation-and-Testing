#!/usr/bin/env python3
"""
Experiment #8815: 6h Donchian breakout + weekly pivot direction + volume confirmation.
Hypothesis: Weekly pivot levels provide strong institutional support/resistance. 
Breakouts above weekly R1 or below weekly S1 with volume confirmation capture institutional moves.
Weekly trend filter (price vs weekly EMA50) ensures alignment with multi-week momentum.
Targets 100-200 total trades over 4 years (25-50/year) for statistical validity while managing fee risk.
"""

from mtf_data import get_ftf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8815_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_EMA_PERIOD = 50
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

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points"""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=WEEKLY_EMA_PERIOD, adjust=False, min_periods=WEEKLY_EMA_PERIOD).mean().values
    
    # Price relative to weekly EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1w > ema_1w, 1, 
                     np.where(close_1w < ema_1w, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1w, price_vs_ema)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points for each week
    pivot_vals = np.full(len(close_1w), np.nan)
    r1_vals = np.full(len(close_1w), np.nan)
    s1_vals = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_1w[i], low_1w[i], close_1w[i])
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        s1_vals[i] = s1
    
    # Align pivot points to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_vals)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
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
        
        # Determine market bias from weekly EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # weekly price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # weekly price below EMA50
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Weekly pivot breakout conditions
        pivot_breakout_long = close[i] > r1_aligned[i-1]  # Break above weekly R1
        pivot_breakout_short = close[i] < s1_aligned[i-1]  # Break below weekly S1
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: require both Donchian and pivot breakout with volume
        long_entry = bull_bias and long_breakout and pivot_breakout_long and volume_confirmed
        short_entry = bear_bias and short_breakout and pivot_breakout_short and volume_confirmed
        
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