#!/usr/bin/env python3
"""
Experiment #8155: 6-hour 1-week pivot-based reversal with 1-day volume confirmation.
Hypothesis: Price rejection at weekly resistance (R3/R4) or support (S3/S4) with 
volume exhaustion (volume < 0.5x 1d MA) on 6h timeframe captures reversals in both 
bull and bear markets. Uses 1d volume to filter for low-volume exhaustion moves 
that often precede reversals, working across market regimes.
"""

from mtf_data import get_ath_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8155_6h_1w_pivot_1d_vol_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # days for weekly pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 0.5  # volume < 50% of MA indicates exhaustion
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, then support/resistance levels"""
    P = (high + low + close) / 3.0
    R1 = 2*P - low
    S1 = 2*P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    R3 = high + 2*(P - low)
    S3 = low - 2*(high - P)
    R4 = R3 + (high - low)
    S4 = S3 - (high - low)
    return P, R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1w for pivots, 1d for volume
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points using prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    P, R1, R2, R3, R4, S1, S2, S3, S4 = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Align pivot levels to 6t timeframe (shifted by 1 week for no look-ahead)
    P_aligned = align_htf_to_ltf(prices, df_1w, P)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Calculate 1d volume moving average for exhaustion filter
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(len(df_1w), len(df_1d), ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(P_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
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
        
        # Volume exhaustion: volume < 50% of 1d MA indicates selling/buying climax
        volume_exhausted = volume[i] < (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1d_aligned[i]) else False
        
        # Reversal signals at weekly pivot extremes
        # Short rejection at resistance (R3/R4) with volume exhaustion
        resistance_rejection = (
            (close[i] < R3_aligned[i]) and 
            (high[i] >= R3_aligned[i]) and 
            volume_exhausted
        ) or (
            (close[i] < R4_aligned[i]) and 
            (high[i] >= R4_aligned[i]) and 
            volume_exhausted
        )
        
        # Long rejection at support (S3/S4) with volume exhaustion
        support_rejection = (
            (close[i] > S3_aligned[i]) and 
            (low[i] <= S3_aligned[i]) and 
            volume_exhausted
        ) or (
            (close[i] > S4_aligned[i]) and 
            (low[i] <= S4_aligned[i]) and 
            volume_exhausted
        )
        
        # Generate signals
        if position == 0:
            if resistance_rejection:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif support_rejection:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals