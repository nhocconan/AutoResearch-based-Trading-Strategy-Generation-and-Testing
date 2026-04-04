#!/usr/bin/env python3
"""
exp_6571_6h_donchian20_1d_pivot_vol_v2
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
Weekly pivots (from prior week) provide stronger institutional levels than daily. 
In bull markets: buy breakouts above weekly R1 with volume.
In bear markets: sell breakdowns below weekly S1 with volume.
In ranging markets: fade extremes at weekly R2/S2 with volume confirmation.
Uses 6h primary timeframe targeting 50-150 total trades over 4 years (12-37/year).
Discrete sizing (0.25) minimizes fee churn.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6571_6h_donchian20_1d_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 30  # ~7.5 days

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE - weekly data from 1d timeframe (we'll resample to weekly ourselves)
    # But per rules, we must use get_htf_data with actual timeframes
    # So we'll use 1d and calculate weekly pivots from it
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivots from daily data
    # Group into weeks (starting Monday) and get weekly OHLC
    # We'll calculate pivots using the prior week's complete data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly OHLC using 5-day periods (approximation for weekly)
    # More accurate: use actual week grouping but we'll use 5-day rollback for simplicity
    # Since we align later, we need values aligned to each 6h bar
    # We'll calculate rolling weekly high/low/close over 5 days
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    
    # Weekly Camarilla-like levels (using standard pivot multipliers)
    # R4 = pivot + (high-low) * 1.1/2
    # R3 = pivot + (high-low) * 1.1/4
    # R2 = pivot + (high-low) * 1.1/6
    # R1 = pivot + (high-low) * 1.1/12
    # S1 = pivot - (high-low) * 1.1/12
    # S2 = pivot - (high-low) * 1.1/6
    # S3 = pivot - (high-low) * 1.1/4
    # S4 = pivot - (high-low) * 1.1/2
    weekly_r4 = weekly_pivot + weekly_range * 1.1 / 2
    weekly_r3 = weekly_pivot + weekly_range * 1.1 / 4
    weekly_r2 = weekly_pivot + weekly_range * 1.1 / 6
    weekly_r1 = weekly_pivot + weekly_range * 1.1 / 12
    weekly_s1 = weekly_pivot - weekly_range * 1.1 / 12
    weekly_s2 = weekly_pivot - weekly_range * 1.1 / 6
    weekly_s3 = weekly_pivot - weekly_range * 1.1 / 4
    weekly_s4 = weekly_pivot - weekly_range * 1.1 / 2
    
    # Align to LTF (6h) with shift(1) for completed bars only
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, 5) + 1  # 5 for weekly calc
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_aligned[i]
        price_below_s1 = close[i] < weekly_s1_aligned[i]
        price_above_r2 = close[i] > weekly_r2_aligned[i]
        price_below_s2 = close[i] < weekly_s2_aligned[i]
        price_between_r1_s1 = (close[i] >= weekly_s1_aligned[i]) & (close[i] <= weekly_r1_aligned[i])
        
        # Volume confirmation
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Long conditions:
        # 1. Break above Donchian HIGH with volume
        # 2. In bullish bias (above weekly R1) OR fading from extreme (below S2 but above S3)
        long_breakout = close[i] > donchian_high[i-1]
        long_continuation = price_above_r1  # Bullish bias
        long_fade = price_below_s2 and close[i] > weekly_s3_aligned[i]  # Fade from S2 but above S3
        
        # Short conditions:
        # 1. Break below Donchian LOW with volume
        # 2. In bearish bias (below weekly S1) OR fading from extreme (above R2 but below R3)
        short_breakout = close[i] < donchian_low[i-1]
        short_continuation = price_below_s1  # Bearish bias
        short_fade = price_above_r2 and close[i] < weekly_r3_aligned[i]  # Fade from R2 but below R3
        
        # Exit conditions: time-based exit OR Donchian midpoint reversal
        if position == 1:  # long position
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if (long_breakout or long_fade) and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif (short_breakout or short_fade) and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals