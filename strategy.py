#!/usr/bin/env python3
"""
Experiment #8667: 6h Donchian breakout + 1d pivot direction + volume confirmation.
Hypothesis: 6h timeframe captures medium-term trends with lower frequency than 1h/4h. Using 1d pivot levels (S1/S2/S3/S4/R1/R2/R3/R4) provides institutional reference points. Breakouts above R3 or below S3 with volume confirmation indicate strong momentum, while bounces at S1/R1 offer mean reversion opportunities in ranging markets. This should work in both bull and bear regimes by adapting to pivot-based support/resistance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8667_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Previous day's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, S1=2P-H, R1=2P-L, etc."""
    pivot = (high + low + close) / 3.0
    s1 = 2 * pivot - high
    r1 = 2 * pivot - low
    s2 = pivot - (high - low)
    r2 = pivot + (high - low)
    s3 = low - 2 * (high - pivot)
    r3 = high + 2 * (pivot - low)
    s4 = s3 - (high - low)
    r4 = r3 + (high - low)
    return pivot, s1, s2, s3, s4, r1, r2, r3, r4

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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day
    pivot_vals = np.zeros(len(high_1d))
    s1_vals = np.zeros(len(high_1d))
    s2_vals = np.zeros(len(high_1d))
    s3_vals = np.zeros(len(high_1d))
    s4_vals = np.zeros(len(high_1d))
    r1_vals = np.zeros(len(high_1d))
    r2_vals = np.zeros(len(high_1d))
    r3_vals = np.zeros(len(high_1d))
    r4_vals = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        p, s1, s2, s3, s4, r1, r2, r3, r4 = calculate_pivot(high_1d[i], low_1d[i], close_1d[i])
        pivot_vals[i] = p
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
        s4_vals[i] = s4
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        r4_vals[i] = r4
    
    # Align to 6h timeframe (previous day's pivots)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)  # R3
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_vals)
    
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
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]):
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
        
        # Determine pivot levels for current day
        s1 = s1_aligned[i]
        s2 = s2_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r1 = r1_aligned[i]
        r2 = r2_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: Breakout at S3/R3 with volume, or bounce at S1/R1
        long_breakout_s3 = long_breakout and close[i] > s3 and volume_confirmed
        short_breakout_s3 = short_breakout and close[i] < s3 and volume_confirmed
        long_bounce_s1 = close[i] <= s1 and close[i] > s2 and volume_confirmed  # Bounce at S1
        short_bounce_r1 = close[i] >= r1 and close[i] < r2 and volume_confirmed  # Bounce at R1
        
        # Generate signals
        if position == 0:
            if long_breakout_s3 or long_bounce_s1:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakout_s3 or short_bounce_r1:
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
</x>