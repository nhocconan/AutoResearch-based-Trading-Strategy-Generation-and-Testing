#!/usr/bin/env python3
"""
Experiment #8487: 6h Donchian breakout + daily pivot direction + volume confirmation.
Hypothesis: 6h timeframe balances trade frequency and trend capture. Daily pivot levels
provide institutional reference points (R3/S3 for fade, R4/S4 for breakout). Volume
confirms institutional participation. Works in bull/bear via pivot-based bias.
Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8487_6h_donchian20_daily_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
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
    """Calculate classic pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L), R3=H+2(P-L), S3=L-2(H-P)"""
    P = (high + low + close) / 3.0
    R1 = 2 * P - low
    S1 = 2 * P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    R3 = high + 2 * (P - low)
    S3 = low - 2 * (high - P)
    R4 = R3 + (high - low)
    S4 = S3 - (high - low)
    return P, R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: daily for pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day
    P_vals, R1_vals, R2_vals, R3_vals, R4_vals, S1_vals, S2_vals, S3_vals, S4_vals = \
        calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Align pivots to 6h timeframe (use previous day's pivots for lookback)
    P_aligned = align_htf_to_ltf(prices, df_1d, P_vals)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_vals)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_vals)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4_vals)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4_vals)
    
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
        if np.isnan(P_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]):
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
        
        # Determine bias from daily pivots: above R3 = bullish bias, below S3 = bearish bias
        bull_bias = close[i] > R3_aligned[i]   # Price above daily R3
        bear_bias = close[i] < S3_aligned[i]   # Price below daily S3
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: fade at R3/S3, breakout at R4/S4
        # Fade logic: if price rejects R3/S3 with volume, go opposite
        # Breakout logic: if price breaks R4/S4 with volume, continue
        fade_long = bull_bias and close[i] < R3_aligned[i] and close[i] > S3_aligned[i] and \
                    close[i] < close[i-1] and volume_confirmed  # Rejecting R3, going long
        fade_short = bear_bias and close[i] > S3_aligned[i] and close[i] < R3_aligned[i] and \
                     close[i] > close[i-1] and volume_confirmed  # Rejecting S3, going short
        breakout_long = close[i] > R4_aligned[i] and volume_confirmed
        breakout_short = close[i] < S4_aligned[i] and volume_confirmed
        
        long_entry = fade_long or breakout_long
        short_entry = fade_short or breakout_short
        
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